#!/usr/bin/env python3
"""
Merge LoRA adapters into base model for vLLM deployment.

This script merges LoRA adapters trained with PEFT/LLaMA Factory into the base
Qwen3-VL model, creating a standalone model that can be loaded directly by vLLM.

Usage:
    python scripts/merge_lora.py \
        --base_model Qwen/Qwen3-VL-8B-Instruct \
        --lora_path outputs/qwen3vl-base-sft \
        --output_path outputs/qwen3vl-base-sft-merged

    # Or with defaults:
    python scripts/merge_lora.py
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor


def merge_lora_to_base(
    base_model_path: str,
    lora_path: str,
    output_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """
    Merge LoRA adapters into base model and save.
    
    Args:
        base_model_path: Path or HuggingFace ID of base model
        lora_path: Path to LoRA adapter checkpoint
        output_path: Path to save merged model
        torch_dtype: Dtype for model weights
    """
    print(f"=" * 60)
    print("LoRA Merge Tool for Qwen3-VL")
    print(f"=" * 60)
    print(f"  Base model:  {base_model_path}")
    print(f"  LoRA path:   {lora_path}")
    print(f"  Output path: {output_path}")
    print(f"  Dtype:       {torch_dtype}")
    print(f"=" * 60)
    
    # Validate paths
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA path not found: {lora_path}")
    
    adapter_config = os.path.join(lora_path, "adapter_config.json")
    if not os.path.exists(adapter_config):
        raise FileNotFoundError(f"No adapter_config.json found in {lora_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load base model
    print("\n[1/4] Loading base model...")
    model = AutoModelForVision2Seq.from_pretrained(
        base_model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="cpu",  # Load to CPU for merging
    )
    print(f"  Base model loaded: {model.__class__.__name__}")
    
    # Load LoRA adapters
    print("\n[2/4] Loading LoRA adapters...")
    model = PeftModel.from_pretrained(model, lora_path)
    print(f"  LoRA adapters loaded from {lora_path}")
    
    # Merge and unload
    print("\n[3/4] Merging LoRA into base model...")
    model = model.merge_and_unload()
    print("  Merge complete!")
    
    # Save merged model
    print(f"\n[4/4] Saving merged model to {output_path}...")
    model.save_pretrained(output_path, safe_serialization=True)
    print("  Model saved!")
    
    # Save processor/tokenizer
    print("\nSaving processor and tokenizer...")
    try:
        # Try loading from LoRA path first (may have custom tokenizer)
        processor = AutoProcessor.from_pretrained(lora_path, trust_remote_code=True)
    except Exception:
        # Fall back to base model
        processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    
    processor.save_pretrained(output_path)
    print("  Processor saved!")
    
    # Verify output
    output_files = list(Path(output_path).glob("*"))
    print(f"\nOutput directory contains {len(output_files)} files:")
    for f in sorted(output_files)[:10]:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)" if size_mb > 1 else f"  - {f.name}")
    
    print(f"\n{'=' * 60}")
    print("Merge complete!")
    print(f"Merged model saved to: {output_path}")
    print(f"\nTo use with vLLM:")
    print(f"  python -m vllm.entrypoints.openai.api_server \\")
    print(f"      --model {output_path} \\")
    print(f"      --port 8000 --trust-remote-code")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters into base model for vLLM deployment"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Base model path or HuggingFace ID"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default="outputs/qwen3vl-base-sft",
        help="Path to LoRA adapter checkpoint"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/qwen3vl-base-sft-merged",
        help="Path to save merged model"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model dtype"
    )
    
    args = parser.parse_args()
    
    # Parse dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]
    
    merge_lora_to_base(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        output_path=args.output_path,
        torch_dtype=torch_dtype,
    )


if __name__ == "__main__":
    main()
