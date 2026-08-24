"""NF4 (bitsandbytes 4-bit) loader for the LTX-2.5 Gemma-4 text encoder.

Loads the self-contained bnb-NF4 text-encoder .safetensors built by
build_gemma_te_nf4.py (te_quant="bnb-nf4"). GPU-only. The gemma4_unified model is
assembled by ltx-core's builder (meta), then the backbone layers are reconstructed
as bnb Linear4bit via QuantState.from_dict; kept-bf16 parts (projections, embed,
norm, vision) are loaded through the standard sd_ops remap.
"""
from __future__ import annotations

import torch  # noqa: E402


def _is_nf4_te_file(path: str) -> bool:
    """True if `path` is a bnb-NF4 text encoder (marker or quant_map keys present)."""
    import safetensors
    try:
        with safetensors.safe_open(path, framework="pt") as f:
            if (f.metadata() or {}).get("te_quant") == "bnb-nf4":
                return True
            return any(k.endswith(".weight.quant_map") for k in f.keys())
    except Exception:
        return False


def _build_meta_te(path: str):
    from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.text_encoders.gemma import GemmaTextEncoderConfigurator, get_gemma_ops

    sd_ops, mod_ops = get_gemma_ops(path)
    builder = SingleGPUModelBuilder(
        model_path=(str(path),),
        model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(str(path)),
        model_sd_ops=sd_ops,
        module_ops=mod_ops,
    )
    metadata = SafetensorsModelStateDictLoader().metadata(path)
    return builder.meta_model(metadata, mod_ops), sd_ops, mod_ops


def _fill_te_bnb_block(layer, sd, device, dtype=torch.bfloat16):
    """Reconstruct one backbone layer's bnb-NF4 linears + plain norms from its flat sd."""
    import bitsandbytes as bnb  # noqa: E402
    from bitsandbytes.functional import QuantState  # noqa: E402

    quant_paths = sorted({k[: -len(".weight.quant_map")]
                          for k in sd if k.endswith(".weight.quant_map")})
    if not quant_paths:
        layer.load_state_dict(sd, strict=False, assign=True)
        return
    plain = {k: v for k, v in sd.items() if not k.startswith(tuple(f"{q}." for q in quant_paths))}
    layer.load_state_dict(plain, strict=False, assign=True)
    for qp in quant_paths:
        parts = qp.split(".")
        parent = layer
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf = parts[-1]
        w_data = sd[f"{qp}.weight"]
        qs_dict = {k: v for k, v in sd.items() if k.startswith(f"{qp}.weight.")}
        qs = QuantState.from_dict(qs_dict, device)
        out_f, in_f = qs.shape
        has_bias = f"{qp}.bias" in sd
        pw = bnb.nn.Params4bit(data=w_data, quant_state=qs, bnb_quantized=True,
                               blocksize=qs.blocksize, quant_type=qs.quant_type,
                               quant_storage=torch.uint8)
        lin = bnb.nn.Linear4bit(in_f, out_f, bias=has_bias, compute_dtype=dtype,
                                quant_type=qs.quant_type, compress_statistics=True)
        lin.weight = pw
        if has_bias:
            lin.bias.data = sd[f"{qp}.bias"].to(device)
        lin = lin.to(device)
        if isinstance(parent, torch.nn.ModuleList) or isinstance(parent, torch.nn.Sequential):
            parent[int(leaf)] = lin
        else:
            setattr(parent, leaf, lin)


def _find_layers(model):
    """Locate the backbone `layers` ModuleList wherever it lives in the model."""
    for name, mod in model.named_modules():
        if name.endswith(".language_model.layers") or (
                name.endswith(".layers") and type(mod).__name__ == "ModuleList"):
            return mod
    raise RuntimeError("could not locate text-encoder backbone layers")


def _load_te_plain(model, sd_ops, file_keys, f, device, dtype=torch.bfloat16):
    """Load kept-bf16 parts (projections/embed/norm/vision) via the sd_ops remap."""
    sd = {}
    for k in file_keys:
        if k.startswith("model.layers."):
            continue
        if k == "tokenizer_json" or k.startswith("hf_asset__"):
            continue
        remapped = sd_ops.apply_to_key(k) if sd_ops is not None else k
        if remapped is None:
            continue
        sd[remapped] = f.get_tensor(k).to(dtype).to(device)
    if sd:
        model.load_state_dict(sd, strict=False, assign=True)
    # tie lm_head -> embed_tokens (locate the language_model dynamically)
    for name, mod in model.named_modules():
        if name.endswith(".language_model"):
            if hasattr(mod, "lm_head") and hasattr(mod, "embed_tokens"):
                try:
                    mod.lm_head.weight.data = mod.embed_tokens.weight.data
                except Exception:
                    pass
            break


def load_gemma_nf4(path, device, dtype=torch.bfloat16):
    """Load the bnb-NF4 text encoder .safetensors onto `device`. GPU-only."""
    import safetensors  # noqa: E402

    model, sd_ops, _ = _build_meta_te(path)
    dev = torch.device(device)
    with safetensors.safe_open(path, framework="pt", device=str(dev)) as f:
        file_keys = set(f.keys())
        idxs = [int(k.split(".")[2]) for k in file_keys
                if k.startswith("model.layers.") and ".mlp.gate_proj.weight" in k]
        n_layers = max(idxs) + 1 if idxs else 0
        layers = _find_layers(model)
        for li in range(n_layers):
            pfx = f"model.layers.{li}."
            sub = {k[len(pfx):]: f.get_tensor(k) for k in file_keys if k.startswith(pfx)}
            _fill_te_bnb_block(layers[li], sub, dev, dtype)
        _load_te_plain(model, sd_ops, file_keys, f, dev, dtype)
    return model
