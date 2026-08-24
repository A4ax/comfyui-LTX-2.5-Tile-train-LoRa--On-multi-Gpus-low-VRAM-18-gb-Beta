"""Quanto-qint2 (2-bit) loader for the LTX-2.5 Gemma-4 text encoder.

Loads the self-contained qint2 text-encoder .safetensors built by
build_gemma_te_int2.py (te_quant="qint2"), plus the qint2 feature-extractor
projections for the embeddings processor. GPU-only. The gemma4_unified model is
assembled by ltx-core's builder (meta), then backbone layers + projections are
reconstructed via quantize(qint2)+freeze+load_state_dict.
"""
from __future__ import annotations

import torch  # noqa: E402
from optimum.quanto import freeze, quantize, qint2  # noqa: E402


def _is_qint2_te_file(path: str) -> bool:
    """True if `path` is a qint2 text encoder (te_quant marker or flattened qint2 keys)."""
    import safetensors
    try:
        with safetensors.safe_open(path, framework="pt") as f:
            if (f.metadata() or {}).get("te_quant") == "qint2":
                return True
            return any(".weight._data._data" in k for k in f.keys())
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
    return builder.meta_model(metadata, mod_ops), sd_ops


def _quantize_and_load(module, sd, device, dtype=torch.bfloat16):
    module.to_empty(device=device)
    module.to(dtype)
    quantize(module, weights=qint2)
    freeze(module)
    module.load_state_dict(sd, strict=False, assign=True)
    del sd
    torch.cuda.synchronize(device)


def _find_layers(model):
    for name, mod in model.named_modules():
        if name.endswith(".language_model.layers") or (
                name.endswith(".layers") and type(mod).__name__ == "ModuleList"):
            return mod
    raise RuntimeError("could not locate text-encoder backbone layers")


def _load_te_plain(model, sd_ops, file_keys, f, device, dtype=torch.bfloat16):
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
    for name, mod in model.named_modules():
        if name.endswith(".language_model"):
            if hasattr(mod, "lm_head") and hasattr(mod, "embed_tokens"):
                try:
                    mod.lm_head.weight.data = mod.embed_tokens.weight.data
                except Exception:
                    pass
            break


def load_gemma_int2(path, device, dtype=torch.bfloat16):
    """Load the qint2 text encoder .safetensors onto `device`. GPU-only."""
    import safetensors  # noqa: E402

    model, sd_ops = _build_meta_te(path)
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
            _quantize_and_load(layers[li], sub, dev, dtype)
        _load_te_plain(model, sd_ops, file_keys, f, dev, dtype)
    return model


def load_embeddings_processor_qint2(checkpoint_path, gemma_model_path, device, dtype=torch.bfloat16):
    """Load the embeddings processor where the feature-extractor projections are qint2.

    The connectors + EP transformer config live in the connector-bearing checkpoint
    (`checkpoint_path[0]`, e.g. embeddings_processor_bf16.safetensors), not in the bnb
    transformer or the packed qint2 TE file. The qint2 feature-extractor projections are
    read from the TE file.
    """
    import safetensors  # noqa: E402
    from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.text_encoders.gemma import (
        EMBEDDINGS_PROCESSOR_KEY_OPS, EmbeddingsProcessorConfigurator,
    )

    torch_device = torch.device(device)
    paths = (tuple(str(p) for p in checkpoint_path)
             if isinstance(checkpoint_path, (list, tuple)) else str(checkpoint_path))
    path_list = list(paths) if isinstance(paths, tuple) else [paths]

    loader = SafetensorsModelStateDictLoader()
    builder = SingleGPUModelBuilder(
        model_path=paths,
        model_class_configurator=EmbeddingsProcessorConfigurator.with_gemma_model_path(str(gemma_model_path)),
        model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
    )

    # Build the EP from the metadata that carries the transformer config (connector dims
    # + FeatureExtractorV2) and gemma_source_checkpoint. The qint2 TE file has no 'config'.
    metadata = loader.metadata(path_list[0])
    if not metadata.get("config", {}).get("transformer"):
        raise ValueError(
            f"Embeddings-processor checkpoint {path_list[0]!r} carries no transformer config. "
            "Pass the embeddings-processor .safetensors (embeddings_processor_bf16.safetensors) first."
        )
    processor = builder.meta_model(metadata, ())

    # feature extractor qint2 (from the packed TE file)
    with safetensors.safe_open(str(gemma_model_path), framework="pt", device=str(torch_device)) as f:
        fe_keys = [k for k in f.keys() if k.startswith("text_embedding_projection.")]
        fe_sd = {k[len("text_embedding_projection."):]: f.get_tensor(k) for k in fe_keys}
        fe = processor.feature_extractor
        fe.to_empty(device=torch_device)
        fe.to(dtype)
        quantize(fe, weights=qint2)
        freeze(fe)
        fe.load_state_dict(fe_sd, strict=False, assign=True)

    # connectors from the connector-bearing file via the builder's loader; exclude
    # feature_extractor.* so the qint2 FE is preserved.
    connector_sd = builder.load_sd(path_list, builder.registry, torch_device,
                                   EMBEDDINGS_PROCESSOR_KEY_OPS).sd
    connector_sd = {k: v for k, v in connector_sd.items() if not k.startswith("feature_extractor.")}
    if connector_sd:
        processor.load_state_dict(connector_sd, strict=False, assign=True)
    return processor
