#!/usr/bin/env python3
"""
Detailed benchmark entrypoint for the Triton track.

This script keeps the existing benchmark files untouched while aligning
the example baseline configuration with benchmark_student.py:
    Linear.BACKEND = "cublas"
    MLP.FUSED = False
    EncoderMLP.FUSED = False

It also adds an explicit configurable warmup stage before detailed profiling.
"""

import argparse
import importlib
import os
import sys

from benchmark_detailed import (
    detailed_profile_torch,
    print_summary,
    profile_attention_ops_torch,
    profile_linear_ops_torch,
    run_nsys_profile,
)
from benchmark_student import load_test_audio, prepare_inputs_torch


TRITON_MODULES = [
    "weight_loader",
    "model",
    "layers",
    "attention",
    "rope",
    "conv",
    "decode_attention",
]


def clear_cached_modules():
    """Clear dynamically imported Triton track modules."""
    for mod_name in list(sys.modules.keys()):
        if mod_name in TRITON_MODULES:
            del sys.modules[mod_name]


def apply_example_baseline_config(folder_name: str):
    """Apply the same baseline configuration used by benchmark_student.py."""
    if "example" not in folder_name.lower():
        return

    print("Applying baseline configuration (example)...")
    layers = importlib.import_module("layers")
    #layers.Linear.BACKEND = "triton" #"cublas"
    layers.MLP.FUSED = True
    if hasattr(layers, "EncoderMLP"):
        layers.EncoderMLP.FUSED = False


def load_triton_model(folder_name: str):
    """Load a Triton track model after applying baseline overrides."""
    print(f"\nLoading model from {folder_name}...")
    from weight_loader import load_model_from_hf

    return load_model_from_hf("zai-org/GLM-ASR-Nano-2512")


def warmup_triton_path(model, input_features, input_ids, num_warmup: int):
    """Warm up the same execution path used by detailed profiling."""
    import torch

    if num_warmup <= 0:
        return

    print(f"\nWarmup ({num_warmup} runs)...")
    embed_tokens = model.text_decoder.embed_tokens

    for idx in range(num_warmup):
        with torch.no_grad():
            audio_features = model.audio_encoder(input_features)
            projected = model.multi_modal_projector(audio_features)
            text_embeds = embed_tokens(input_ids)

            audio_mask = input_ids == 59260
            combined_embeds = text_embeds.clone()
            if torch.any(audio_mask):
                audio_positions = torch.where(audio_mask[0])[0]
                num_audio_tokens = int(audio_positions.numel())
                if num_audio_tokens <= projected.shape[1]:
                    combined_embeds[0, audio_positions[:projected.shape[1]]] = projected[0, :num_audio_tokens]

            hidden_states = model.text_decoder(inputs_embeds=combined_embeds)
            logits = model.lm_head(hidden_states[:, -1:, :])
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            next_embed = embed_tokens(next_token)
            _ = model.text_decoder(inputs_embeds=next_embed)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        print(f"  Warmup {idx + 1}/{num_warmup} complete")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detailed operator profiling for the Triton track"
    )
    parser.add_argument(
        "folder",
        type=str,
        nargs="?",
        default="glm_asr_triton_example",
        help="Triton folder name to benchmark",
    )
    parser.add_argument("--audio", type=str, help="Path to test audio file")
    parser.add_argument(
        "--runs", type=int, default=3, help="Number of profiling runs"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of explicit warmup runs before profiling",
    )
    parser.add_argument(
        "--nsys", action="store_true", help="Run Nsight Systems profiling"
    )
    parser.add_argument(
        "--attention-only",
        action="store_true",
        help="Only profile attention operations",
    )
    parser.add_argument(
        "--linear-only",
        action="store_true",
        help="Only profile linear/GEMM operations",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=256,
        help="Sequence length for micro-benchmarks",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if "triton" not in args.folder.lower():
        raise ValueError(
            f"This entrypoint is only for Triton track folders, got: {args.folder}"
        )

    print("=" * 70)
    print("GLM-ASR Detailed Operator Profiling (Triton Track)")
    print("=" * 70)

    if args.nsys:
        run_nsys_profile(args.folder, args.audio)
        return 0

    if args.attention_only:
        profile_attention_ops_torch(seq_len=args.seq_len, num_runs=args.runs)
        return 0

    if args.linear_only:
        profile_linear_ops_torch(seq_len=args.seq_len, num_runs=args.runs)
        return 0

    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, args.folder)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    print("\nLoading test audio...")
    audio_array, _, duration = load_test_audio(args.audio)
    print(f"Audio duration: {duration:.2f}s")

    sys.path.insert(0, folder_path)
    try:
        clear_cached_modules()
        apply_example_baseline_config(args.folder)
        model, processor = load_triton_model(args.folder)

        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_features, input_ids, input_features_mask = prepare_inputs_torch(
            audio_array, processor, device
        )

        print(f"Input features shape: {tuple(input_features.shape)}")
        print(f"Input IDs shape: {tuple(input_ids.shape)}")

        warmup_triton_path(model, input_features, input_ids, args.warmup)

        component_results = detailed_profile_torch(
            model,
            input_features,
            input_ids,
            input_features_mask,
            num_runs=args.runs,
        )
        attention_results = profile_attention_ops_torch(
            seq_len=args.seq_len, num_runs=args.runs
        )
        linear_results = profile_linear_ops_torch(
            seq_len=args.seq_len, num_runs=args.runs
        )

        print_summary(component_results, attention_results, linear_results)
    finally:
        if folder_path in sys.path:
            sys.path.remove(folder_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
