"""Generate a voice clip via the LTX T2A pipeline (woman/girl or man/boy).

Used by the Dataset node when the user leaves the voice upload empty: it
synthesizes a voice clip which then becomes the audio training data, so the
resulting LoRA produces that voice. Shells out to the engine venv.

Run:  engine_python generate_voice.py --transformer <int4> --text-encoder <gemma> \\
        --audio-vae <vae> --out <dir> --voice woman --frames 49 --fps 24 [--duration-head]
"""
import argparse
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import engine_env  # noqa: E402
engine_env.setup_paths()
from msvc_env import apply_msvc_env  # noqa: E402
apply_msvc_env()

import torch  # noqa: E402

from ltx_pipelines.t2a_one_stage import T2AOneStagePipeline  # noqa: E402
from ltx_pipelines.utils.model_paths import ModelPaths  # noqa: E402
from ltx_pipelines.utils.media_io import encode_audio  # noqa: E402
from ltx_core.components.guiders import MultiModalGuiderParams  # noqa: E402

VOICE_PROMPTS = {
    "auto": "A clear, natural human voice speaking.",
    "woman / girl": "A woman's voice, warm and expressive, speaking clearly.",
    "man / boy": "A man's voice, deep and calm, speaking clearly.",
}
NEGATIVE = ("robotic voice, echo, background noise, off-sync audio, "
            "repetitive speech, jittery, distortion, mumbling, whispering")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transformer", required=True)
    ap.add_argument("--text-encoder", required=True)
    ap.add_argument("--audio-vae", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--voice", default="auto", choices=list(VOICE_PROMPTS))
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--inference-steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--duration-head", default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_wav = os.path.join(args.out_dir, "generated_voice.wav")

    prompt = VOICE_PROMPTS.get(args.voice, VOICE_PROMPTS["auto"])
    device = torch.device(args.device)

    paths = ModelPaths.from_split(
        transformer_path=args.transformer,
        text_encoder_path=args.text_encoder,
        audio_vae_path=args.audio_vae,
        duration_head_path=args.duration_head,
    )

    print(f"[generate_voice] prompt: {prompt}", flush=True)
    pipe = T2AOneStagePipeline(
        model_paths=paths,
        loras=(),
        device=device,
    )
    audio = pipe(
        prompt=prompt,
        negative_prompt=NEGATIVE,
        seed=args.seed,
        num_frames=args.frames,
        frame_rate=args.fps,
        num_inference_steps=args.inference_steps,
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
            modality_scale=1.0, skip_step=0, stg_blocks=[29],
        ),
        enhance_prompt=False,
    )
    encode_audio(audio=audio, output_path=out_wav)
    print(f"[generate_voice] DONE -> {out_wav} ({os.path.getsize(out_wav)/1e6:.1f}MB)", flush=True)


if __name__ == "__main__":
    main()
