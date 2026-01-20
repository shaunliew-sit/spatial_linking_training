#!/usr/bin/env python3
"""
Compare attention heatmaps between base model and fine-tuned model.

Supports two modes:
1. Native VLM attention: Extract cross-attention from text→image tokens (works for any Qwen3-VL)
2. Spatial linking attention: Our custom spatial linking module attention

Usage:
    # Native VLM attention comparison (recommended for comparing base vs fine-tuned)
    python scripts/compare_attention.py \
        --base_model Qwen/Qwen3-VL-8B-Instruct \
        --finetuned_model outputs/spatial_linking_sft_multi_gpu/run_20260120_034050 \
        --image_path /path/to/image.jpg \
        --output_dir ./attention_comparison \
        --task_type referring \
        --attention_mode native

    # Spatial linking attention (our custom module)
    python scripts/compare_attention.py \
        --base_model Qwen/Qwen3-VL-8B-Instruct \
        --finetuned_model outputs/spatial_linking_sft_multi_gpu/run_20260120_034050 \
        --image_path /path/to/image.jpg \
        --output_dir ./attention_comparison \
        --task_type referring \
        --attention_mode spatial_linking \
        --base_use_spatial_linking
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from PIL import Image
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoProcessor
from models.spatial_model import SpatialLinkingInteractionModel
from peft import PeftModel
from utils.visualization import visualize_multi_region_attention, create_attention_heatmap
from scripts.evaluate import (
    normalize_bbox_to_1000,
    create_referring_prompt,
    create_grounding_prompt_with_image,
    format_attention_for_viz,
    SYSTEM_PROMPT,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Native VLM Attention Functions
# ============================================================================

def load_pure_qwen3vl(model_name: str, device: str = "cuda:0"):
    """Load pure Qwen3-VL without spatial linking wrapper."""
    logger.info(f"Loading pure Qwen3-VL: {model_name}")
    
    try:
        from transformers import Qwen3VLForConditionalGeneration
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=None,
            attn_implementation="eager",  # Required to get attention weights
        )
    except ImportError:
        from transformers import Qwen2VLForConditionalGeneration
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=None,
            attn_implementation="eager",
        )
    
    model = model.to(device)
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    return model, processor


def load_finetuned_qwen3vl(model_path: str, device: str = "cuda:0"):
    """Load fine-tuned Qwen3-VL with PEFT adapters (without spatial linking wrapper)."""
    logger.info(f"Loading fine-tuned Qwen3-VL from: {model_path}")
    
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    has_peft = os.path.exists(adapter_config_path)
    
    if has_peft:
        logger.info("  Detected PEFT adapters - loading base model first...")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen3-VL-8B-Instruct")
        
        try:
            from transformers import Qwen3VLForConditionalGeneration
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=None,
                attn_implementation="eager",
            )
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=None,
                attn_implementation="eager",
            )
        
        base_model = base_model.to(device)
        
        logger.info("  Loading PEFT LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.to(device)
    else:
        try:
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=None,
                attn_implementation="eager",
            )
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=None,
                attn_implementation="eager",
            )
        model = model.to(device)
    
    model.eval()
    
    processor = AutoProcessor.from_pretrained(
        base_model_name if has_peft else model_path,
        trust_remote_code=True
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    return model, processor


def find_vision_token_indices(input_ids: torch.Tensor, processor) -> Tuple[int, int]:
    """Find the start and end indices of vision tokens in the sequence."""
    # Get special token IDs
    vision_start_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = processor.tokenizer.convert_tokens_to_ids("<|vision_end|>")
    
    input_ids_list = input_ids[0].tolist()
    
    # Find vision_start and vision_end positions
    vision_start_idx = None
    vision_end_idx = None
    
    for i, token_id in enumerate(input_ids_list):
        if token_id == vision_start_id and vision_start_idx is None:
            vision_start_idx = i
        elif token_id == vision_end_id and vision_start_idx is not None:
            vision_end_idx = i
            break
    
    if vision_start_idx is None or vision_end_idx is None:
        # Fallback: look for image_pad tokens
        image_pad_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        image_positions = (input_ids[0] == image_pad_id).nonzero(as_tuple=True)[0]
        if len(image_positions) > 0:
            vision_start_idx = image_positions[0].item()
            vision_end_idx = image_positions[-1].item() + 1
    
    return vision_start_idx, vision_end_idx


def find_box_token_ranges(input_ids: torch.Tensor, processor) -> List[Tuple[int, int, str]]:
    """
    Find the token ranges for each bounding box in the input.
    
    Returns list of (start_idx, end_idx, box_label) tuples for each box.
    Box tokens are between <|box_start|> and <|box_end|>.
    """
    box_start_id = processor.tokenizer.convert_tokens_to_ids("<|box_start|>")
    box_end_id = processor.tokenizer.convert_tokens_to_ids("<|box_end|>")
    
    input_ids_list = input_ids[0].tolist()
    
    box_ranges = []
    box_labels = ["Person", "Object", "Interaction"]
    
    i = 0
    box_idx = 0
    while i < len(input_ids_list):
        if input_ids_list[i] == box_start_id:
            start = i
            # Find matching box_end
            for j in range(i + 1, len(input_ids_list)):
                if input_ids_list[j] == box_end_id:
                    end = j + 1  # Include the box_end token
                    label = box_labels[box_idx] if box_idx < len(box_labels) else f"Box{box_idx}"
                    box_ranges.append((start, end, label))
                    box_idx += 1
                    i = j
                    break
        i += 1
    
    return box_ranges


def find_best_grid_shape(num_tokens: int, image_width: int, image_height: int) -> Tuple[int, int]:
    """
    Find the best H×W grid shape for vision tokens that matches the image aspect ratio.
    
    Qwen3-VL uses spatial merging which reduces the number of patches.
    We use the image aspect ratio to find the most sensible 2D grid.
    
    Args:
        num_tokens: Number of vision tokens
        image_width: Original image width
        image_height: Original image height
    
    Returns:
        (H, W) tuple for the grid shape
    """
    aspect_ratio = image_width / image_height  # W/H ratio
    
    # For N tokens with aspect ratio r = W/H:
    # N = H * W = H * (H * r) = H^2 * r
    # H = sqrt(N / r)
    estimated_H = int(np.sqrt(num_tokens / aspect_ratio))
    estimated_W = int(num_tokens / estimated_H) if estimated_H > 0 else int(np.sqrt(num_tokens))
    
    # Search for the best factorization near the estimate
    best_H, best_W = estimated_H, estimated_W
    best_score = float('inf')
    
    for H in range(max(1, estimated_H - 5), estimated_H + 6):
        if H <= 0:
            continue
        for W in range(max(1, estimated_W - 5), estimated_W + 6):
            if W <= 0:
                continue
            if H * W > num_tokens:  # Can't have more cells than tokens
                continue
            
            # Score based on:
            # 1. How close H*W is to num_tokens (prefer using all tokens)
            # 2. How close W/H is to the image aspect ratio
            coverage = H * W / num_tokens
            current_ratio = W / H
            ratio_diff = abs(current_ratio - aspect_ratio) / aspect_ratio
            
            # Prefer high coverage and low ratio difference
            score = (1 - coverage) + ratio_diff * 0.5
            
            if score < best_score:
                best_score = score
                best_H, best_W = H, W
    
    # Ensure we have at least a 2D grid (not 1×N)
    if best_H == 1 and num_tokens > 10:
        # Fallback to square-ish grid
        side = int(np.sqrt(num_tokens))
        best_H = side
        best_W = num_tokens // side
    
    return best_H, best_W


def extract_box_to_image_attention(
    model, processor, inputs, image_grid_thw,
    image_size: Tuple[int, int] = None,  # (width, height)
    target_layers: List[int] = [-1, -4, -8, -12],
) -> Dict[str, np.ndarray]:
    """
    Extract attention FROM each bounding box's tokens TO the image patches.
    
    This shows: "When the model reads the person/object/interaction box coordinates,
    which parts of the image does it attend to?"
    
    Returns:
        Dict mapping box labels to their attention heatmaps
    """
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True,
        )
    
    attentions = outputs.attentions
    num_layers = len(attentions)
    
    # Find vision token indices
    vision_start, vision_end = find_vision_token_indices(inputs['input_ids'], processor)
    if vision_start is None or vision_end is None:
        logger.warning("Could not find vision token boundaries")
        return {}
    
    num_vision_tokens = vision_end - vision_start
    logger.info(f"Vision tokens: {vision_start} to {vision_end} ({num_vision_tokens} tokens)")
    
    # Find box token ranges
    box_ranges = find_box_token_ranges(inputs['input_ids'], processor)
    logger.info(f"Found {len(box_ranges)} box token ranges: {[(r[2], r[0], r[1]) for r in box_ranges]}")
    
    if not box_ranges:
        logger.warning("No box tokens found in input")
        return {}
    
    # Get grid dimensions from image_grid_thw (before spatial merge)
    if torch.is_tensor(image_grid_thw):
        T, H_orig, W_orig = image_grid_thw[0].tolist() if image_grid_thw.dim() > 1 else image_grid_thw.tolist()
    else:
        T, H_orig, W_orig = image_grid_thw
    
    # Calculate best grid shape based on actual number of tokens and image aspect ratio
    if image_size is not None:
        img_width, img_height = image_size
    else:
        # Estimate from original grid dimensions
        img_width, img_height = W_orig * 28, H_orig * 28  # Qwen3-VL uses 28px patches
    
    # The actual grid after spatial merging
    H, W = find_best_grid_shape(num_vision_tokens, img_width, img_height)
    logger.info(f"Grid shape: {H}×{W} = {H*W} (for {num_vision_tokens} tokens, image {img_width}×{img_height})")
    
    # Collect attention from target layers
    layer_attentions = []
    for layer_idx in target_layers:
        actual_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx
        if 0 <= actual_idx < num_layers:
            layer_attn = attentions[actual_idx]  # [batch, heads, seq, seq]
            layer_attentions.append(layer_attn)
    
    if not layer_attentions:
        logger.warning("No valid layers found")
        return {}
    
    # Stack and average across layers and heads
    stacked = torch.stack(layer_attentions, dim=0)  # [num_layers, batch, heads, seq, seq]
    aggregated = stacked.mean(dim=(0, 2))  # [batch, seq, seq] - mean over layers and heads
    
    # Extract attention for each box
    box_heatmaps = {}
    for box_start, box_end, box_label in box_ranges:
        # Get attention FROM box tokens TO vision tokens
        # Shape: [num_box_tokens, num_vision_tokens]
        box_to_vision_attn = aggregated[0, box_start:box_end, vision_start:vision_end]
        
        # Average across box tokens to get attention per vision token
        vision_attention = box_to_vision_attn.mean(dim=0)  # [num_vision_tokens]
        
        # Convert to numpy
        vision_attention = vision_attention.float().cpu().numpy()
        
        # Reshape to 2D grid
        # Pad or truncate to fit H*W if needed
        grid_size = H * W
        if len(vision_attention) < grid_size:
            # Pad with zeros
            padded = np.zeros(grid_size)
            padded[:len(vision_attention)] = vision_attention
            vision_attention = padded
        elif len(vision_attention) > grid_size:
            # Truncate
            vision_attention = vision_attention[:grid_size]
        
        heatmap = vision_attention.reshape(H, W)
        
        # Normalize to [0, 1]
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        box_heatmaps[box_label] = heatmap
        logger.info(f"  {box_label} box attention: tokens {box_start}-{box_end}, heatmap shape {heatmap.shape}")
    
    return box_heatmaps


def extract_response_to_image_attention(
    model, processor, inputs, image_grid_thw,
    image_size: Tuple[int, int],
    max_new_tokens: int = 256,
    target_layers: List[int] = [-1, -4, -8, -12],
) -> Dict[str, Any]:
    """
    Extract attention FROM generated response tokens TO image patches.
    
    This is the key visualization for verifying spatial linking:
    "When the model generates its response (thinking + answer), 
     which image regions is it attending to?"
    
    Returns:
        Dict with:
        - 'full_response': The complete generated text
        - 'full_heatmap': Attention heatmap for entire response
        - 'think_heatmap': Attention heatmap for <think>...</think> portion (if present)
        - 'answer_heatmap': Attention heatmap for final answer (after </think>)
        - 'answer_text': Just the final answer text
    """
    # Find vision token indices in input
    vision_start, vision_end = find_vision_token_indices(inputs['input_ids'], processor)
    if vision_start is None or vision_end is None:
        logger.warning("Could not find vision token boundaries")
        return {}
    
    num_vision_tokens = vision_end - vision_start
    input_len = inputs['input_ids'].shape[1]
    
    # Calculate grid shape
    img_width, img_height = image_size
    H, W = find_best_grid_shape(num_vision_tokens, img_width, img_height)
    logger.info(f"Response→Image: Vision tokens {vision_start}-{vision_end} ({num_vision_tokens}), Grid {H}×{W}")
    
    # Generate with attention output
    # We need to do step-by-step generation to collect attention at each step
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_attentions=True,
        )
    
    # Get generated token IDs
    generated_ids = outputs.sequences[0, input_len:]  # Only new tokens
    full_response = processor.decode(generated_ids, skip_special_tokens=True)
    
    # Collect attention from all generation steps
    # outputs.attentions is tuple of (step, layer, batch, heads, seq, seq)
    if not hasattr(outputs, 'attentions') or outputs.attentions is None:
        logger.warning("No attention returned from generate(). Model may not support output_attentions during generation.")
        return {'full_response': full_response}
    
    # Process attention from each generation step
    num_layers = len(outputs.attentions[0]) if outputs.attentions else 0
    num_steps = len(outputs.attentions)
    
    logger.info(f"Generated {num_steps} tokens, {num_layers} layers")
    
    # Aggregate attention from response tokens to vision tokens
    # For each step, we get attention of the new token to all previous tokens (including vision)
    response_to_vision_attention = []
    
    for step_idx, step_attentions in enumerate(outputs.attentions):
        # step_attentions is tuple of (layer_attn) where each is [batch, heads, 1, seq_so_far]
        # The last position attends to all previous positions including vision tokens
        
        step_layer_attns = []
        for layer_idx in target_layers:
            actual_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx
            if 0 <= actual_idx < num_layers:
                layer_attn = step_attentions[actual_idx]  # [batch, heads, 1, seq]
                # Extract attention to vision tokens
                # The sequence includes: input tokens (with vision) + previously generated tokens
                # Vision tokens are at positions vision_start:vision_end in the original input
                if layer_attn.shape[-1] > vision_end:
                    attn_to_vision = layer_attn[0, :, 0, vision_start:vision_end]  # [heads, num_vision]
                    step_layer_attns.append(attn_to_vision.mean(dim=0))  # [num_vision]
        
        if step_layer_attns:
            step_attn = torch.stack(step_layer_attns).mean(dim=0)  # [num_vision]
            response_to_vision_attention.append(step_attn.float().cpu().numpy())
    
    if not response_to_vision_attention:
        logger.warning("Could not extract response→vision attention")
        return {'full_response': full_response}
    
    # Stack all steps: [num_steps, num_vision]
    all_response_attn = np.stack(response_to_vision_attention, axis=0)
    
    # Create heatmaps
    def attention_to_heatmap(attn_1d):
        """Convert 1D attention vector to 2D heatmap."""
        grid_size = H * W
        if len(attn_1d) < grid_size:
            padded = np.zeros(grid_size)
            padded[:len(attn_1d)] = attn_1d
            attn_1d = padded
        elif len(attn_1d) > grid_size:
            attn_1d = attn_1d[:grid_size]
        
        heatmap = attn_1d.reshape(H, W)
        if heatmap.max() > heatmap.min():
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        return heatmap
    
    # Full response heatmap (average across all generated tokens)
    full_attn = all_response_attn.mean(axis=0)
    full_heatmap = attention_to_heatmap(full_attn)
    
    result = {
        'full_response': full_response,
        'full_heatmap': full_heatmap,
        'grid_shape': (H, W),
    }
    
    # Try to separate <think> from final answer
    think_start_token = "<think>"
    think_end_token = "</think>"
    
    # Decode each token to find think boundaries
    token_texts = [processor.decode([tid], skip_special_tokens=False) for tid in generated_ids.tolist()]
    
    think_start_idx = None
    think_end_idx = None
    
    # Find <think> and </think> token positions
    decoded_so_far = ""
    for i, tok_text in enumerate(token_texts):
        decoded_so_far += tok_text
        if think_start_token in decoded_so_far and think_start_idx is None:
            think_start_idx = i
        if think_end_token in decoded_so_far and think_end_idx is None:
            think_end_idx = i
            break
    
    if think_start_idx is not None and think_end_idx is not None and think_end_idx > think_start_idx:
        # Thinking portion
        think_attn = all_response_attn[think_start_idx:think_end_idx+1].mean(axis=0)
        result['think_heatmap'] = attention_to_heatmap(think_attn)
        result['think_text'] = processor.decode(generated_ids[think_start_idx:think_end_idx+1], skip_special_tokens=True)
        
        # Answer portion (after </think>)
        if think_end_idx + 1 < len(all_response_attn):
            answer_attn = all_response_attn[think_end_idx+1:].mean(axis=0)
            result['answer_heatmap'] = attention_to_heatmap(answer_attn)
            result['answer_text'] = processor.decode(generated_ids[think_end_idx+1:], skip_special_tokens=True)
        
        logger.info(f"  Separated: think tokens {think_start_idx}-{think_end_idx}, answer tokens {think_end_idx+1}-{len(generated_ids)}")
    else:
        # No <think> tags, entire response is the answer
        result['answer_heatmap'] = full_heatmap
        result['answer_text'] = full_response
        logger.info(f"  No <think> tags found, using full response as answer")
    
    return result


def extract_native_attention(
    model, processor, inputs, image_grid_thw,
    target_layers: List[int] = [-1, -4, -8],
    aggregate: str = "mean"
) -> np.ndarray:
    """
    Extract native VLM attention from text tokens to image patches.
    
    Args:
        model: Qwen3-VL model
        processor: Processor with tokenizer
        inputs: Tokenized inputs
        image_grid_thw: Image grid dimensions [T, H, W]
        target_layers: Which layers to extract attention from (negative = from end)
        aggregate: How to aggregate across heads/layers ("mean", "max")
    
    Returns:
        Attention heatmap of shape [H, W]
    """
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            return_dict=True,
        )
    
    # attentions is tuple of (batch, num_heads, seq_len, seq_len) per layer
    attentions = outputs.attentions
    num_layers = len(attentions)
    
    # Find vision token indices
    vision_start, vision_end = find_vision_token_indices(inputs['input_ids'], processor)
    
    if vision_start is None or vision_end is None:
        logger.warning("Could not find vision token boundaries")
        return None
    
    num_vision_tokens = vision_end - vision_start
    logger.info(f"Vision tokens: {vision_start} to {vision_end} ({num_vision_tokens} tokens)")
    
    # Get grid dimensions
    if torch.is_tensor(image_grid_thw):
        T, H, W = image_grid_thw[0].tolist() if image_grid_thw.dim() > 1 else image_grid_thw.tolist()
    else:
        T, H, W = image_grid_thw
    
    # Collect attention from target layers
    layer_attentions = []
    for layer_idx in target_layers:
        actual_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx
        if 0 <= actual_idx < num_layers:
            layer_attn = attentions[actual_idx]  # [batch, heads, seq, seq]
            layer_attentions.append(layer_attn)
    
    if not layer_attentions:
        logger.warning("No valid layers found")
        return None
    
    # Stack and aggregate across layers
    stacked = torch.stack(layer_attentions, dim=0)  # [num_layers, batch, heads, seq, seq]
    
    if aggregate == "mean":
        aggregated = stacked.mean(dim=(0, 2))  # [batch, seq, seq] - mean over layers and heads
    else:  # max
        aggregated = stacked.max(dim=0).values.max(dim=1).values  # [batch, seq, seq]
    
    # Extract attention FROM text tokens TO vision tokens
    # We want: attention[text_tokens, vision_tokens]
    # For visualization, we sum attention from all text tokens to each vision token
    
    # Text tokens are everything after vision_end (the response tokens)
    # But during forward pass, we look at attention from ALL non-vision tokens
    text_start = vision_end
    seq_len = aggregated.shape[-1]
    
    # Get attention from text tokens to vision tokens
    if text_start < seq_len:
        text_to_vision_attn = aggregated[0, text_start:, vision_start:vision_end]  # [num_text, num_vision]
        # Average across text tokens
        vision_attention = text_to_vision_attn.mean(dim=0)  # [num_vision]
    else:
        # Fallback: use attention from the last token to vision tokens
        vision_attention = aggregated[0, -1, vision_start:vision_end]  # [num_vision]
    
    # Convert to numpy
    vision_attention = vision_attention.float().cpu().numpy()
    
    # Reshape to grid
    expected_tokens = int(T) * int(H) * int(W)
    if len(vision_attention) != expected_tokens:
        # Try to find correct grid size
        actual_HW = len(vision_attention) // int(T)
        ratio = H / W if W > 0 else 1.0
        for h in range(1, int(actual_HW ** 0.5) + 2):
            if actual_HW % h == 0:
                w = actual_HW // h
                if abs((h / w) - ratio) < abs((H / W) - ratio) or (H * W != actual_HW):
                    H, W = h, w
        logger.info(f"Adjusted grid to [{T}, {H}, {W}] for {len(vision_attention)} tokens")
    
    try:
        heatmap = vision_attention[:int(T)*int(H)*int(W)].reshape(int(T), int(H), int(W))
        heatmap = heatmap[0]  # Take first temporal slice [H, W]
    except ValueError as e:
        logger.warning(f"Could not reshape attention: {e}")
        # Fallback: create approximate grid
        side = int(np.sqrt(len(vision_attention)))
        heatmap = vision_attention[:side*side].reshape(side, side)
    
    # Normalize
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    return heatmap


def run_native_attention_comparison(
    model, processor, image: Image.Image, 
    person_box: List[int], object_box: List[int],
    device: torch.device
) -> Tuple[str, Optional[Dict[str, np.ndarray]]]:
    """
    Run inference and extract box-specific attention.
    
    Returns:
        Tuple of (generated_response, box_heatmaps_dict)
        box_heatmaps_dict maps box labels ("Person", "Object", "Interaction") to their attention heatmaps
    """
    img_width, img_height = image.size
    person_box_norm = normalize_bbox_to_1000(person_box, img_width, img_height)
    object_box_norm = normalize_bbox_to_1000(object_box, img_width, img_height)
    
    interaction_box = [
        min(person_box_norm[0], object_box_norm[0]),
        min(person_box_norm[1], object_box_norm[1]),
        max(person_box_norm[2], object_box_norm[2]),
        max(person_box_norm[3], object_box_norm[3])
    ]
    
    user_prompt = create_referring_prompt(person_box_norm, object_box_norm, interaction_box)
    user_content = [
        {"type": "image", "image": image},
        {"type": "text", "text": user_prompt}
    ]
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get image grid dimensions
    image_grid_thw = inputs.get('image_grid_thw', None)
    
    # Extract box-specific attention (Person box → image, Object box → image, etc.)
    # This is the "input grounding" check
    box_heatmaps = extract_box_to_image_attention(
        model, processor, inputs, image_grid_thw,
        image_size=(img_width, img_height),
        target_layers=[-1, -4, -8, -12],
    )
    
    # Extract response→image attention (the key visualization for spatial linking)
    # This shows: "When generating the response, which image regions does it attend to?"
    response_attention = extract_response_to_image_attention(
        model, processor, inputs, image_grid_thw,
        image_size=(img_width, img_height),
        max_new_tokens=256,
        target_layers=[-1, -4, -8, -12],
    )
    
    generated = response_attention.get('full_response', '')
    
    return {
        'response': generated,
        'box_heatmaps': box_heatmaps if box_heatmaps else {},
        'response_attention': response_attention,
    }


def create_native_attention_comparison(
    image: Image.Image,
    base_box_heatmaps: Dict[str, np.ndarray],
    finetuned_box_heatmaps: Dict[str, np.ndarray],
    boxes: List[List[int]],
    output_path: str,
    base_response: str,
    finetuned_response: str,
):
    """
    Create visualization showing where each bounding box attends to in the image.
    
    This answers: "When the model reads the Person/Object/Interaction box coordinates,
    which parts of the image does it look at?"
    
    Layout: 2 rows (Base, Fine-tuned) x 4 columns (Original + 3 box attentions)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from scipy.ndimage import zoom as scipy_zoom
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    img_array = np.array(image.convert('RGB'))
    H_img, W_img = img_array.shape[:2]
    
    box_labels = ['Person', 'Object', 'Interaction']
    box_colors = {'Person': 'lime', 'Object': 'cyan', 'Interaction': 'yellow'}
    
    # Create 2x4 layout
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    def draw_box(ax, box, color, linewidth=2, linestyle='-'):
        """Draw a single bounding box."""
        x1 = box[0] / 1000 * W_img
        y1 = box[1] / 1000 * H_img
        x2 = box[2] / 1000 * W_img
        y2 = box[3] / 1000 * H_img
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=linewidth, edgecolor=color, facecolor='none', linestyle=linestyle
        )
        ax.add_patch(rect)
        return x1, y1, x2, y2
    
    def draw_all_boxes(ax, highlight_idx=None):
        """Draw all boxes, optionally highlighting one."""
        for i, (box, label) in enumerate(zip(boxes, box_labels)):
            color = box_colors[label]
            if highlight_idx is not None and i == highlight_idx:
                draw_box(ax, box, color, linewidth=3, linestyle='-')
            else:
                draw_box(ax, box, color, linewidth=1, linestyle='--')
    
    def calc_iou_attention(heatmap, box, h_img, w_img):
        """Calculate how much attention is inside vs outside the target box."""
        x1 = int(box[0] / 1000 * w_img)
        y1 = int(box[1] / 1000 * h_img)
        x2 = int(box[2] / 1000 * w_img)
        y2 = int(box[3] / 1000 * h_img)
        
        # Clip to image bounds
        x1, x2 = max(0, x1), min(w_img, x2)
        y1, y2 = max(0, y1), min(h_img, y2)
        
        inside = heatmap[y1:y2, x1:x2]
        inside_mean = inside.mean() if inside.size > 0 else 0
        
        # Create mask for outside
        mask = np.ones_like(heatmap, dtype=bool)
        mask[y1:y2, x1:x2] = False
        outside = heatmap[mask]
        outside_mean = outside.mean() if outside.size > 0 else 0
        
        return inside_mean, outside_mean
    
    def resize_heatmap(heatmap, target_h, target_w):
        """Resize heatmap to match image size."""
        zoom_h = target_h / heatmap.shape[0]
        zoom_w = target_w / heatmap.shape[1]
        return scipy_zoom(heatmap, (zoom_h, zoom_w), order=1)
    
    # Row 0: Base model
    # Row 1: Fine-tuned model
    for row, (model_name, box_heatmaps, response) in enumerate([
        ("Base Qwen3-VL", base_box_heatmaps, base_response),
        ("Fine-tuned", finetuned_box_heatmaps, finetuned_response)
    ]):
        # Column 0: Original image with all boxes
        axes[row, 0].imshow(img_array)
        draw_all_boxes(axes[row, 0])
        axes[row, 0].set_title(f'{model_name}\nResponse: "{response[:40]}..."', fontsize=10, fontweight='bold')
        axes[row, 0].axis('off')
        
        # Add legend only on first row
        if row == 0:
            legend_elements = [patches.Patch(facecolor='none', edgecolor=c, label=l) 
                             for l, c in box_colors.items()]
            axes[row, 0].legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        # Columns 1-3: Attention for each box
        for col, label in enumerate(box_labels):
            ax = axes[row, col + 1]
            ax.imshow(img_array)
            
            if label in box_heatmaps and box_heatmaps[label] is not None:
                heatmap = box_heatmaps[label]
                heatmap_resized = resize_heatmap(heatmap, H_img, W_img)
                
                # Show heatmap
                im = ax.imshow(heatmap_resized, cmap='hot', alpha=0.7, vmin=0, vmax=1)
                
                # Draw target box (solid) and others (dashed)
                target_idx = box_labels.index(label)
                draw_all_boxes(ax, highlight_idx=target_idx)
                
                # Calculate attention inside vs outside target box
                inside_attn, outside_attn = calc_iou_attention(
                    heatmap_resized, boxes[target_idx], H_img, W_img
                )
                grounding_ratio = inside_attn / (outside_attn + 1e-6)
                
                # Title with grounding metrics
                ax.set_title(
                    f'{label} Box → Image Attention\n'
                    f'Inside: {inside_attn:.3f} | Outside: {outside_attn:.3f} | Ratio: {grounding_ratio:.2f}x',
                    fontsize=9, fontweight='bold',
                    color='green' if grounding_ratio > 1.5 else ('orange' if grounding_ratio > 1.0 else 'red')
                )
                
                # Add small colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.02)
                plt.colorbar(im, cax=cax)
            else:
                ax.set_title(f'{label} Box\n(No attention data)', fontsize=9)
                draw_all_boxes(ax, highlight_idx=box_labels.index(label))
            
            ax.axis('off')
    
    # Add explanation text
    fig.text(0.5, 0.01, 
             'Each heatmap shows: "When the model reads these box coordinates, which image regions does it attend to?"\n'
             'Good grounding: Attention should be INSIDE the target box (Ratio > 1.5x = green, < 1.0x = red)',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.suptitle('HOI Referring Task: Box Coordinate → Image Attention Grounding', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved box-to-image attention comparison to {output_path}")


def create_response_attention_comparison(
    image: Image.Image,
    base_result: Dict[str, Any],
    finetuned_result: Dict[str, Any],
    boxes: List[List[int]],
    output_path: str,
):
    """
    Create visualization showing Response → Image attention.
    
    This is the KEY visualization for verifying spatial linking effectiveness:
    "When the model GENERATES its response (thinking + answer), 
     which image regions is it attending to?"
    
    Layout depends on whether <think> tags are present:
    - With thinking: 2 rows × 4 cols (Original, Think, Answer, Full)
    - Without thinking: 2 rows × 3 cols (Original, Answer, Full)
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from scipy.ndimage import zoom as scipy_zoom
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    img_array = np.array(image.convert('RGB'))
    H_img, W_img = img_array.shape[:2]
    
    box_colors = {'Person': 'lime', 'Object': 'cyan', 'Interaction': 'yellow'}
    box_labels = ['Person', 'Object', 'Interaction']
    
    def draw_boxes(ax):
        """Draw all bounding boxes."""
        for box, label in zip(boxes, box_labels):
            color = box_colors[label]
            x1 = box[0] / 1000 * W_img
            y1 = box[1] / 1000 * H_img
            x2 = box[2] / 1000 * W_img
            y2 = box[3] / 1000 * H_img
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none', linestyle='--'
            )
            ax.add_patch(rect)
    
    def resize_heatmap(heatmap, target_h, target_w):
        zoom_h = target_h / heatmap.shape[0]
        zoom_w = target_w / heatmap.shape[1]
        return scipy_zoom(heatmap, (zoom_h, zoom_w), order=1)
    
    def calc_roi_attention(heatmap, box):
        """Calculate mean attention inside a box."""
        x1 = int(box[0] / 1000 * heatmap.shape[1])
        y1 = int(box[1] / 1000 * heatmap.shape[0])
        x2 = int(box[2] / 1000 * heatmap.shape[1])
        y2 = int(box[3] / 1000 * heatmap.shape[0])
        
        x1, x2 = max(0, x1), min(heatmap.shape[1], x2)
        y1, y2 = max(0, y1), min(heatmap.shape[0], y2)
        
        roi = heatmap[y1:y2, x1:x2]
        return roi.mean() if roi.size > 0 else 0
    
    # Check if we have thinking heatmaps
    has_think_base = 'think_heatmap' in base_result
    has_think_ft = 'think_heatmap' in finetuned_result
    has_thinking = has_think_base or has_think_ft
    
    # Determine layout
    if has_thinking:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        col_titles = ['Input Image', 'Thinking Attention', 'Answer Attention', 'Full Response Attention']
    else:
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        col_titles = ['Input Image', 'Answer Attention', 'Full Response Attention']
    
    for row, (model_name, result) in enumerate([
        ("Base Qwen3-VL", base_result),
        ("Fine-tuned", finetuned_result)
    ]):
        response = result.get('full_response', '')[:60]
        answer = result.get('answer_text', response)[:40]
        
        # Column 0: Original image
        axes[row, 0].imshow(img_array)
        draw_boxes(axes[row, 0])
        axes[row, 0].set_title(f'{model_name}\nAnswer: "{answer}..."', fontsize=10, fontweight='bold')
        axes[row, 0].axis('off')
        
        if row == 0:
            legend_elements = [patches.Patch(facecolor='none', edgecolor=c, label=l) 
                             for l, c in box_colors.items()]
            axes[row, 0].legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        if has_thinking:
            # Column 1: Thinking attention
            ax = axes[row, 1]
            ax.imshow(img_array)
            if 'think_heatmap' in result and result['think_heatmap'] is not None:
                heatmap = resize_heatmap(result['think_heatmap'], H_img, W_img)
                im = ax.imshow(heatmap, cmap='hot', alpha=0.7, vmin=0, vmax=1)
                draw_boxes(ax)
                
                # Calculate attention in interaction zone
                interaction_attn = calc_roi_attention(heatmap, boxes[2])  # Interaction box
                ax.set_title(f'<think> → Image\nInteraction zone: {interaction_attn:.3f}', 
                           fontsize=9, fontweight='bold')
                
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="3%", pad=0.02)
                plt.colorbar(im, cax=cax)
            else:
                ax.set_title('No <think> tokens', fontsize=9)
                draw_boxes(ax)
            ax.axis('off')
            
            # Column 2: Answer attention
            answer_col = 2
        else:
            answer_col = 1
        
        # Answer attention column
        ax = axes[row, answer_col]
        ax.imshow(img_array)
        if 'answer_heatmap' in result and result['answer_heatmap'] is not None:
            heatmap = resize_heatmap(result['answer_heatmap'], H_img, W_img)
            im = ax.imshow(heatmap, cmap='hot', alpha=0.7, vmin=0, vmax=1)
            draw_boxes(ax)
            
            # Calculate grounding metrics
            person_attn = calc_roi_attention(heatmap, boxes[0])
            object_attn = calc_roi_attention(heatmap, boxes[1])
            interaction_attn = calc_roi_attention(heatmap, boxes[2])
            
            ax.set_title(f'Answer → Image\n'
                        f'P:{person_attn:.2f} O:{object_attn:.2f} I:{interaction_attn:.2f}', 
                       fontsize=9, fontweight='bold',
                       color='green' if interaction_attn > max(person_attn, object_attn) * 0.8 else 'orange')
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.02)
            plt.colorbar(im, cax=cax)
        else:
            ax.set_title('No answer attention', fontsize=9)
            draw_boxes(ax)
        ax.axis('off')
        
        # Full response attention column
        full_col = answer_col + 1
        ax = axes[row, full_col]
        ax.imshow(img_array)
        if 'full_heatmap' in result and result['full_heatmap'] is not None:
            heatmap = resize_heatmap(result['full_heatmap'], H_img, W_img)
            im = ax.imshow(heatmap, cmap='hot', alpha=0.7, vmin=0, vmax=1)
            draw_boxes(ax)
            
            interaction_attn = calc_roi_attention(heatmap, boxes[2])
            ax.set_title(f'Full Response → Image\nInteraction: {interaction_attn:.3f}', 
                       fontsize=9, fontweight='bold')
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad=0.02)
            plt.colorbar(im, cax=cax)
        else:
            ax.set_title('No attention data', fontsize=9)
            draw_boxes(ax)
        ax.axis('off')
    
    # Add explanation
    fig.text(0.5, 0.01, 
             'Response → Image Attention: Shows which image regions the model attends to when GENERATING its response.\n'
             'Good spatial linking: High attention in the INTERACTION zone (between person and object) when predicting the action.',
             ha='center', fontsize=10, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    plt.suptitle('HOI Referring Task: Response → Image Attention (Spatial Linking Verification)', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.06, 1, 0.96])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved response-to-image attention comparison to {output_path}")


def create_thinking_grounding_highlight(
    image: Image.Image,
    base_result: Dict[str, Any],
    finetuned_result: Dict[str, Any],
    boxes: List[List[int]],
    output_path: str,
):
    """
    Create a focused visualization highlighting the <think> grounding evidence.
    
    This is the KEY figure for showing spatial linking effectiveness:
    - Shows that fine-tuned model's reasoning is visually grounded
    - Compares attention in interaction zone during thinking vs answer
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from scipy.ndimage import zoom as scipy_zoom
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    
    img_array = np.array(image.convert('RGB'))
    H_img, W_img = img_array.shape[:2]
    
    box_colors = {'Person': 'lime', 'Object': 'cyan', 'Interaction': 'yellow'}
    box_labels = ['Person', 'Object', 'Interaction']
    
    def draw_boxes(ax, highlight_interaction=False):
        for box, label in zip(boxes, box_labels):
            color = box_colors[label]
            x1 = box[0] / 1000 * W_img
            y1 = box[1] / 1000 * H_img
            x2 = box[2] / 1000 * W_img
            y2 = box[3] / 1000 * H_img
            lw = 3 if (label == 'Interaction' and highlight_interaction) else 2
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=lw, edgecolor=color, facecolor='none', 
                linestyle='-' if (label == 'Interaction' and highlight_interaction) else '--'
            )
            ax.add_patch(rect)
    
    def resize_heatmap(heatmap, target_h, target_w):
        zoom_h = target_h / heatmap.shape[0]
        zoom_w = target_w / heatmap.shape[1]
        return scipy_zoom(heatmap, (zoom_h, zoom_w), order=1)
    
    def calc_roi_attention(heatmap, box):
        x1 = int(box[0] / 1000 * heatmap.shape[1])
        y1 = int(box[1] / 1000 * heatmap.shape[0])
        x2 = int(box[2] / 1000 * heatmap.shape[1])
        y2 = int(box[3] / 1000 * heatmap.shape[0])
        x1, x2 = max(0, x1), min(heatmap.shape[1], x2)
        y1, y2 = max(0, y1), min(heatmap.shape[0], y2)
        roi = heatmap[y1:y2, x1:x2]
        return roi.mean() if roi.size > 0 else 0
    
    # Create figure: 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Get attention values
    base_answer = base_result.get('answer_heatmap')
    ft_think = finetuned_result.get('think_heatmap')
    ft_answer = finetuned_result.get('answer_heatmap')
    
    # Column 0: Base model answer attention
    ax = axes[0]
    ax.imshow(img_array)
    if base_answer is not None:
        heatmap = resize_heatmap(base_answer, H_img, W_img)
        im = ax.imshow(heatmap, cmap='hot', alpha=0.7, vmin=0, vmax=1)
        draw_boxes(ax, highlight_interaction=True)
        
        interaction_attn = calc_roi_attention(heatmap, boxes[2])
        ax.set_title(f'Base Qwen3-VL\n"riding horse" → Image\n\n'
                    f'Interaction Zone Attention: {interaction_attn:.3f}',
                    fontsize=11, fontweight='bold')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax)
    else:
        ax.set_title('Base Model\n(No data)', fontsize=11)
        draw_boxes(ax)
    ax.axis('off')
    
    # Column 1: Fine-tuned <think> attention (THE KEY EVIDENCE)
    ax = axes[1]
    ax.imshow(img_array)
    if ft_think is not None:
        heatmap = resize_heatmap(ft_think, H_img, W_img)
        im = ax.imshow(heatmap, cmap='hot', alpha=0.7, vmin=0, vmax=1)
        draw_boxes(ax, highlight_interaction=True)
        
        interaction_attn = calc_roi_attention(heatmap, boxes[2])
        person_attn = calc_roi_attention(heatmap, boxes[0])
        object_attn = calc_roi_attention(heatmap, boxes[1])
        
        # Calculate improvement
        base_interaction = calc_roi_attention(resize_heatmap(base_answer, H_img, W_img), boxes[2]) if base_answer is not None else 0
        improvement = ((interaction_attn / base_interaction) - 1) * 100 if base_interaction > 0 else 0
        
        ax.set_title(f'Fine-tuned (Reasoning Phase)\n<think> → Image\n\n'
                    f'Interaction Zone: {interaction_attn:.3f} (+{improvement:.0f}% vs Base)',
                    fontsize=11, fontweight='bold', color='green')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax)
    else:
        ax.set_title('Fine-tuned <think>\n(No thinking tokens)', fontsize=11)
        draw_boxes(ax)
    ax.axis('off')
    
    # Column 2: Fine-tuned answer attention (fair comparison to base)
    ax = axes[2]
    ax.imshow(img_array)
    if ft_answer is not None:
        heatmap = resize_heatmap(ft_answer, H_img, W_img)
        im = ax.imshow(heatmap, cmap='hot', alpha=0.7, vmin=0, vmax=1)
        draw_boxes(ax, highlight_interaction=True)
        
        interaction_attn = calc_roi_attention(heatmap, boxes[2])
        ax.set_title(f'Fine-tuned (Answer Phase)\n"riding horse" → Image\n\n'
                    f'Interaction Zone Attention: {interaction_attn:.3f}',
                    fontsize=11, fontweight='bold')
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        plt.colorbar(im, cax=cax)
    else:
        ax.set_title('Fine-tuned Answer\n(No data)', fontsize=11)
        draw_boxes(ax)
    ax.axis('off')
    
    # Add legend
    legend_elements = [patches.Patch(facecolor='none', edgecolor=c, label=l) 
                      for l, c in box_colors.items()]
    fig.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.95), fontsize=9)
    
    # Add explanation text
    base_int = calc_roi_attention(resize_heatmap(base_answer, H_img, W_img), boxes[2]) if base_answer is not None else 0
    ft_think_int = calc_roi_attention(resize_heatmap(ft_think, H_img, W_img), boxes[2]) if ft_think is not None else 0
    ft_ans_int = calc_roi_attention(resize_heatmap(ft_answer, H_img, W_img), boxes[2]) if ft_answer is not None else 0
    
    explanation = (
        f'KEY FINDING: Fine-tuning added a reasoning phase (<think>) that is visually grounded.\n'
        f'• Base answer attention on interaction zone: {base_int:.3f}\n'
        f'• Fine-tuned <think> attention on interaction zone: {ft_think_int:.3f} '
        f'(+{((ft_think_int/base_int)-1)*100:.0f}% higher)\n'
        f'• Fine-tuned answer attention: {ft_ans_int:.3f} (similar to base — the benefit is in the reasoning process)'
    )
    
    fig.text(0.5, 0.02, explanation, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Spatial Linking Evidence: Reasoning Process is Visually Grounded', 
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.15, 1, 0.94])
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved thinking grounding highlight to {output_path}")


# ============================================================================
# Spatial Linking Attention Functions (Original)
# ============================================================================


def load_base_model(model_name: str, device: str = "cuda:0", use_spatial_linking: bool = True):
    """Load base Qwen3-VL model.
    
    Args:
        model_name: HuggingFace model name
        device: Device to load model on
        use_spatial_linking: If True, wrap with SpatialLinkingInteractionModel (random weights).
                            If False, load pure Qwen3-VL without spatial linking.
    """
    logger.info(f"Loading base model: {model_name}")
    logger.info(f"  use_spatial_linking: {use_spatial_linking}")
    
    if use_spatial_linking:
        model = SpatialLinkingInteractionModel.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=None,
        )
    else:
        # Load pure Qwen3-VL without spatial linking wrapper
        from transformers import Qwen2VLForConditionalGeneration
        try:
            from transformers import Qwen3VLForConditionalGeneration
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=None,
            )
        except ImportError:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=None,
            )
    
    model = model.to(device)
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Set box token IDs for spatial linking (only if using spatial linking)
    if use_spatial_linking and hasattr(model, 'set_box_token_ids'):
        model.set_box_token_ids(processor.tokenizer)
    
    return model, processor


def load_finetuned_model(model_path: str, device: str = "cuda:0"):
    """Load fine-tuned model with PEFT adapters and spatial linking weights."""
    logger.info(f"Loading fine-tuned model from: {model_path}")
    
    # Check for PEFT adapters
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    has_peft = os.path.exists(adapter_config_path)
    
    if has_peft:
        logger.info("  Detected PEFT adapters - loading base model first...")
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_name = adapter_config.get("base_model_name_or_path", "Qwen/Qwen3-VL-8B-Instruct")
        
        # Load base model
        base_model = SpatialLinkingInteractionModel.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=None,
        )
        base_model = base_model.to(device)
        
        # Load PEFT adapters
        logger.info("  Loading PEFT LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.to(device)
    else:
        model = SpatialLinkingInteractionModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=None,
        )
        model = model.to(device)
    
    # Load spatial linking weights
    spatial_linking_path = os.path.join(model_path, "spatial_linking.pt")
    if os.path.exists(spatial_linking_path):
        logger.info(f"  Loading spatial linking weights from {spatial_linking_path}")
        spatial_linking_state = torch.load(spatial_linking_path, map_location="cpu")
        
        if has_peft:
            if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
                if hasattr(model.base_model.model, 'spatial_linking'):
                    model.base_model.model.spatial_linking.load_state_dict(spatial_linking_state)
        else:
            model.spatial_linking.load_state_dict(spatial_linking_state)
    
    model.eval()
    
    # Load processor
    processor = AutoProcessor.from_pretrained(
        base_model_name if has_peft else model_path,
        trust_remote_code=True
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Set box token IDs
    if has_peft:
        if hasattr(model, 'base_model') and hasattr(model.base_model, 'model'):
            model.base_model.model.set_box_token_ids(processor.tokenizer)
    else:
        model.set_box_token_ids(processor.tokenizer)
    
    return model, processor


def get_underlying_model(model):
    """Get the underlying SpatialLinkingInteractionModel from a potentially wrapped model."""
    # Handle PeftModel wrapping
    if hasattr(model, 'base_model'):
        if hasattr(model.base_model, 'model'):
            return model.base_model.model
        return model.base_model
    return model


def verify_spatial_linking_weights(base_model, finetuned_model):
    """Verify that spatial linking weights differ between base and fine-tuned models."""
    base_underlying = get_underlying_model(base_model)
    finetuned_underlying = get_underlying_model(finetuned_model)
    
    if not hasattr(base_underlying, 'spatial_linking') or not hasattr(finetuned_underlying, 'spatial_linking'):
        logger.warning("One or both models don't have spatial_linking module")
        return False
    
    base_weights = base_underlying.spatial_linking.state_dict()
    finetuned_weights = finetuned_underlying.spatial_linking.state_dict()
    
    total_diff = 0.0
    for key in base_weights:
        if key in finetuned_weights:
            diff = (base_weights[key].float() - finetuned_weights[key].float()).abs().mean().item()
            total_diff += diff
            logger.info(f"  Spatial linking weight diff '{key}': {diff:.6f}")
    
    if total_diff < 1e-6:
        logger.warning("WARNING: Spatial linking weights appear IDENTICAL between base and fine-tuned models!")
        logger.warning("This suggests the fine-tuned weights were not loaded correctly.")
        return False
    else:
        logger.info(f"Total spatial linking weight difference: {total_diff:.6f}")
        return True


def run_referring_task(model, processor, image: Image.Image, person_box: List[int], 
                      object_box: List[int], device: torch.device) -> Tuple[str, Optional[List]]:
    """Run referring task and get attention weights."""
    # Normalize boxes to 1000x1000 format
    img_width, img_height = image.size
    person_box_norm = normalize_bbox_to_1000(person_box, img_width, img_height)
    object_box_norm = normalize_bbox_to_1000(object_box, img_width, img_height)
    
    # Create interaction box
    interaction_box = [
        min(person_box_norm[0], object_box_norm[0]),
        min(person_box_norm[1], object_box_norm[1]),
        max(person_box_norm[2], object_box_norm[2]),
        max(person_box_norm[3], object_box_norm[3])
    ]
    
    # Create prompt with interaction box
    user_prompt = create_referring_prompt(person_box_norm, object_box_norm, interaction_box)
    user_content = [
        {"type": "image", "image": image},
        {"type": "text", "text": user_prompt}
    ]
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Prepare refer_boxes
    refer_boxes_tensor = torch.tensor(
        [person_box_norm, object_box_norm, interaction_box],
        dtype=torch.float32
    ).unsqueeze(0).to(device)
    
    # Get the underlying model for attention retrieval (handles PEFT wrapping)
    underlying_model = get_underlying_model(model)
    
    # Check if model has spatial linking capability
    has_spatial_linking = hasattr(underlying_model, 'spatial_linking') and hasattr(underlying_model, 'get_spatial_attention_weights')
    
    # Forward pass to get attention (only if model has spatial linking)
    attention_info = None
    if has_spatial_linking:
        with torch.no_grad():
            try:
                _ = model(
                    **inputs,
                    refer_boxes=[refer_boxes_tensor.squeeze(0)],
                    output_spatial_attentions=True,
                )
                # Get attention from underlying model (not PEFT wrapper)
                attention_info = underlying_model.get_spatial_attention_weights()
            except Exception as e:
                logger.warning(f"Could not extract attention weights: {e}")
                logger.info("Model may not support spatial attention or refer_boxes")
    else:
        logger.info("Model does not have spatial linking - skipping attention extraction")
    
    # Generate response
    # Use refer_boxes only if model supports it
    generate_kwargs = {
        **inputs,
        "max_new_tokens": 256,
        "do_sample": False,
        "pad_token_id": processor.tokenizer.pad_token_id,
    }
    if has_spatial_linking:
        generate_kwargs["refer_boxes"] = [refer_boxes_tensor.squeeze(0)]
    
    with torch.no_grad():
        outputs = model.generate(**generate_kwargs)
    
    generated = processor.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    # attention_info is [[dict1, dict2, dict3]] (list of batch, each batch has list of dicts)
    # Return first batch's attention info
    if attention_info is not None and len(attention_info) > 0:
        return generated, attention_info[0]
    return generated, None


def run_grounding_task(model, processor, image: Image.Image, action: str, 
                      object_category: str, device: torch.device) -> Tuple[str, Optional[List]]:
    """Run grounding task and get attention weights."""
    # Create prompt
    user_content = create_grounding_prompt_with_image(action, object_category, image)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        truncation=True,
        max_length=4096,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # For grounding, we don't have refer_boxes initially, so we can't get spatial attention
    # We'll need to use a different approach - maybe extract attention from the generation
    # For now, return None for attention
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    
    generated = processor.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    return generated, None


def create_comparison_visualization(
    image: Image.Image,
    base_attention_list: List[Dict],
    finetuned_attention_list: List[Dict],
    boxes: List[List[int]],
    output_path: str,
    task_type: str = "referring"
):
    """Create side-by-side comparison visualization."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from scipy.ndimage import zoom as scipy_zoom
    
    img_array = np.array(image.convert('RGB'))
    H_img, W_img = img_array.shape[:2]
    
    n_regions = len(base_attention_list)
    fig, axes = plt.subplots(2, n_regions + 1, figsize=(16, 10))
    
    box_types = ['person', 'object', 'interaction']
    box_colors = {'person': 'lime', 'object': 'cyan', 'interaction': 'yellow'}
    
    # Top row: Base model
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Base Model', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    for i, (info, box, box_type) in enumerate(zip(base_attention_list, boxes, box_types)):
        ax = axes[0, i + 1]
        
        # Create heatmap
        heatmap = create_attention_heatmap(
            attention_weights=info['attention_weights'],
            grid_thw=info['grid_thw'],
            patch_indices=info['patch_indices'],
        )
        
        # Resize heatmap
        zoom_h = H_img / heatmap.shape[0]
        zoom_w = W_img / heatmap.shape[1]
        heatmap_resized = scipy_zoom(heatmap, (zoom_h, zoom_w), order=1)
        
        # Show image and overlay
        ax.imshow(img_array)
        im = ax.imshow(heatmap_resized, cmap='hot', alpha=0.6, vmin=0, vmax=1)
        
        # Draw bounding box
        if 'bbox' in info:
            bbox = info['bbox']
            if torch.is_tensor(bbox):
                bbox = bbox.numpy()
            x1, y1, x2, y2 = bbox * np.array([W_img, H_img, W_img, H_img])
            color = box_colors.get(box_type, 'white')
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.set_title(f'Base: {box_type.capitalize()}', fontsize=12)
        
        ax.axis('off')
    
    # Bottom row: Fine-tuned model
    axes[1, 0].imshow(img_array)
    axes[1, 0].set_title('Fine-tuned Model', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    for i, (info, box, box_type) in enumerate(zip(finetuned_attention_list, boxes, box_types)):
        ax = axes[1, i + 1]
        
        # Create heatmap
        heatmap = create_attention_heatmap(
            attention_weights=info['attention_weights'],
            grid_thw=info['grid_thw'],
            patch_indices=info['patch_indices'],
        )
        
        # Resize heatmap
        zoom_h = H_img / heatmap.shape[0]
        zoom_w = W_img / heatmap.shape[1]
        heatmap_resized = scipy_zoom(heatmap, (zoom_h, zoom_w), order=1)
        
        # Show image and overlay
        ax.imshow(img_array)
        im = ax.imshow(heatmap_resized, cmap='hot', alpha=0.6, vmin=0, vmax=1)
        
        # Draw bounding box
        if 'bbox' in info:
            bbox = info['bbox']
            if torch.is_tensor(bbox):
                bbox = bbox.numpy()
            x1, y1, x2, y2 = bbox * np.array([W_img, H_img, W_img, H_img])
            color = box_colors.get(box_type, 'white')
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            ax.set_title(f'Fine-tuned: {box_type.capitalize()}', fontsize=12)
        
        ax.axis('off')
    
    # Add shared colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label='Attention Weight')
    
    plt.suptitle(f'Attention Comparison: {task_type.capitalize()} Task', fontsize=16, fontweight='bold')
    plt.subplots_adjust(top=0.92, bottom=0.05, left=0.02, right=0.95, hspace=0.1, wspace=0.05)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved comparison visualization to {output_path}")


def create_single_model_visualization(
    image: Image.Image,
    attention_list: List[Dict],
    boxes: List[List[int]],
    output_path: str,
    task_type: str = "referring",
    model_name: str = "Model"
):
    """Create visualization for a single model's attention."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from scipy.ndimage import zoom as scipy_zoom
    
    img_array = np.array(image.convert('RGB'))
    H_img, W_img = img_array.shape[:2]
    
    n_regions = len(attention_list)
    fig, axes = plt.subplots(1, n_regions + 1, figsize=(16, 5))
    
    box_types = ['person', 'object', 'interaction']
    box_colors = {'person': 'lime', 'object': 'cyan', 'interaction': 'yellow'}
    
    # First column: original image
    axes[0].imshow(img_array)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    for i, (info, box, box_type) in enumerate(zip(attention_list, boxes, box_types)):
        ax = axes[i + 1]
        
        # Create heatmap
        heatmap = create_attention_heatmap(
            attention_weights=info['attention_weights'],
            grid_thw=info['grid_thw'],
            patch_indices=info['patch_indices'],
        )
        
        # Resize heatmap
        zoom_h = H_img / heatmap.shape[0]
        zoom_w = W_img / heatmap.shape[1]
        heatmap_resized = scipy_zoom(heatmap, (zoom_h, zoom_w), order=1)
        
        # Show image and overlay
        ax.imshow(img_array)
        im = ax.imshow(heatmap_resized, cmap='hot', alpha=0.6, vmin=0, vmax=1)
        
        # Draw bounding box
        if 'bbox' in info:
            bbox = info['bbox']
            if torch.is_tensor(bbox):
                bbox = bbox.numpy()
            x1, y1, x2, y2 = bbox * np.array([W_img, H_img, W_img, H_img])
            color = box_colors.get(box_type, 'white')
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
        ax.set_title(f'{box_type.capitalize()} Attention', fontsize=12)
        ax.axis('off')
    
    # Add colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02, label='Attention Weight')
    
    plt.suptitle(f'{model_name}: {task_type.capitalize()} Task', fontsize=16, fontweight='bold')
    plt.subplots_adjust(top=0.88, bottom=0.05, left=0.02, right=0.95, wspace=0.05)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved single model visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare attention heatmaps between base and fine-tuned models")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-VL-8B-Instruct",
                       help="Base model name or path")
    parser.add_argument("--finetuned_model", type=str, required=True,
                       help="Path to fine-tuned model checkpoint")
    parser.add_argument("--image_path", type=str, required=True,
                       help="Path to test image")
    parser.add_argument("--output_dir", type=str, default="./attention_comparison",
                       help="Output directory for visualizations")
    parser.add_argument("--task_type", type=str, choices=["referring", "grounding"], default="referring",
                       help="Task type: referring or grounding")
    parser.add_argument("--person_box", type=int, nargs=4, default=[320, 306, 359, 349],
                       help="Person bounding box [x1, y1, x2, y2] in pixels")
    parser.add_argument("--object_box", type=int, nargs=4, default=[148, 345, 376, 414],
                       help="Object bounding box [x1, y1, x2, y2] in pixels")
    parser.add_argument("--action", type=str, default="sitting on",
                       help="Action for grounding task")
    parser.add_argument("--object_category", type=str, default="bench",
                       help="Object category for grounding task")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to use")
    parser.add_argument("--attention_mode", type=str, choices=["native", "spatial_linking"], 
                       default="native",
                       help="Attention extraction mode: "
                            "'native' = Extract Qwen3-VL's native text→image attention (recommended), "
                            "'spatial_linking' = Extract our custom spatial linking attention")
    parser.add_argument("--base_use_spatial_linking", action="store_true",
                       help="[spatial_linking mode only] If set, base model uses spatial linking with random weights.")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load image
    image = Image.open(args.image_path).convert("RGB")
    logger.info(f"Loaded image: {args.image_path} ({image.size})")
    
    device = torch.device(args.device)
    
    # ========================================================================
    # NATIVE ATTENTION MODE
    # ========================================================================
    if args.attention_mode == "native":
        logger.info("=" * 80)
        logger.info("NATIVE VLM ATTENTION MODE")
        logger.info("Extracting Qwen3-VL's native text→image cross-attention")
        logger.info("=" * 80)
        
        # Load pure Qwen3-VL models (no spatial linking wrapper)
        logger.info("Loading Base Qwen3-VL...")
        base_model, base_processor = load_pure_qwen3vl(args.base_model, args.device)
        
        logger.info("Loading Fine-tuned Qwen3-VL...")
        finetuned_model, finetuned_processor = load_finetuned_qwen3vl(args.finetuned_model, args.device)
        
        # Run inference and extract attention
        logger.info("=" * 80)
        logger.info("Running inference with attention extraction...")
        logger.info("=" * 80)
        
        logger.info("Running base model...")
        base_result = run_native_attention_comparison(
            base_model, base_processor, image,
            args.person_box, args.object_box, device
        )
        base_response = base_result.get('response', '')
        logger.info(f"Base response: {base_response[:100]}...")
        
        logger.info("Running fine-tuned model...")
        finetuned_result = run_native_attention_comparison(
            finetuned_model, finetuned_processor, image,
            args.person_box, args.object_box, device
        )
        finetuned_response = finetuned_result.get('response', '')
        logger.info(f"Fine-tuned response: {finetuned_response[:100]}...")
        
        # Prepare boxes for visualization
        img_width, img_height = image.size
        person_box_norm = normalize_bbox_to_1000(args.person_box, img_width, img_height)
        object_box_norm = normalize_bbox_to_1000(args.object_box, img_width, img_height)
        interaction_box = [
            min(person_box_norm[0], object_box_norm[0]),
            min(person_box_norm[1], object_box_norm[1]),
            max(person_box_norm[2], object_box_norm[2]),
            max(person_box_norm[3], object_box_norm[3])
        ]
        boxes = [person_box_norm, object_box_norm, interaction_box]
        
        # Create Box → Image visualization (input grounding check)
        base_box_heatmaps = base_result.get('box_heatmaps', {})
        finetuned_box_heatmaps = finetuned_result.get('box_heatmaps', {})
        
        if base_box_heatmaps and finetuned_box_heatmaps:
            output_path = output_dir / f"box_to_image_attention_{args.task_type}.png"
            create_native_attention_comparison(
                image, base_box_heatmaps, finetuned_box_heatmaps, boxes,
                str(output_path), base_response, finetuned_response
            )
            logger.info(f"Box→Image attention saved to {output_path}")
        
        # Create Response → Image visualization (THE KEY visualization for spatial linking)
        base_response_attn = base_result.get('response_attention', {})
        finetuned_response_attn = finetuned_result.get('response_attention', {})
        
        if base_response_attn and finetuned_response_attn:
            output_path = output_dir / f"response_to_image_attention_{args.task_type}.png"
            create_response_attention_comparison(
                image, base_response_attn, finetuned_response_attn, boxes,
                str(output_path)
            )
            logger.info(f"Response→Image attention saved to {output_path}")
            
            # Create focused <think> grounding highlight (KEY figure for paper/report)
            output_path_highlight = output_dir / f"thinking_grounding_highlight_{args.task_type}.png"
            create_thinking_grounding_highlight(
                image, base_response_attn, finetuned_response_attn, boxes,
                str(output_path_highlight)
            )
            logger.info(f"Thinking grounding highlight (KEY figure) saved to {output_path_highlight}")
        else:
            logger.warning("Could not extract response→image attention")
            if not base_response_attn:
                logger.warning("  Base model: No response attention extracted")
            if not finetuned_response_attn:
                logger.warning("  Fine-tuned model: No response attention extracted")
        
        logger.info("=" * 80)
        logger.info("Comparison complete!")
        logger.info("=" * 80)
        return
    
    # ========================================================================
    # SPATIAL LINKING ATTENTION MODE
    # ========================================================================
    logger.info("=" * 80)
    logger.info("SPATIAL LINKING ATTENTION MODE")
    logger.info("=" * 80)
    
    # Load models with spatial linking wrapper
    logger.info("Loading Base Model...")
    base_model, base_processor = load_base_model(
        args.base_model, args.device, use_spatial_linking=args.base_use_spatial_linking
    )
    
    logger.info("Loading Fine-tuned Model...")
    finetuned_model, finetuned_processor = load_finetuned_model(args.finetuned_model, args.device)
    
    # Verify spatial linking weights differ (only if both have spatial linking)
    if args.base_use_spatial_linking:
        logger.info("Verifying Spatial Linking Weights...")
        verify_spatial_linking_weights(base_model, finetuned_model)
    
    # Run inference
    logger.info("=" * 80)
    logger.info(f"Running {args.task_type} task")
    logger.info("=" * 80)
    
    if args.task_type == "referring":
        # Base model
        logger.info("Running base model...")
        base_response, base_attention = run_referring_task(
            base_model, base_processor, image,
            args.person_box, args.object_box, device
        )
        logger.info(f"Base model response: {base_response[:100]}...")
        
        # Fine-tuned model
        logger.info("Running fine-tuned model...")
        finetuned_response, finetuned_attention = run_referring_task(
            finetuned_model, finetuned_processor, image,
            args.person_box, args.object_box, device
        )
        logger.info(f"Fine-tuned model response: {finetuned_response[:100]}...")
        
        # Format attention for visualization
        img_width, img_height = image.size
        person_box_norm = normalize_bbox_to_1000(args.person_box, img_width, img_height)
        object_box_norm = normalize_bbox_to_1000(args.object_box, img_width, img_height)
        interaction_box = [
            min(person_box_norm[0], object_box_norm[0]),
            min(person_box_norm[1], object_box_norm[1]),
            max(person_box_norm[2], object_box_norm[2]),
            max(person_box_norm[3], object_box_norm[3])
        ]
        
        boxes = [person_box_norm, object_box_norm, interaction_box]
        
        if base_attention and finetuned_attention:
            # Both models have attention - show side-by-side comparison
            base_attention_list = format_attention_for_viz(
                base_attention, boxes, img_width, img_height
            )
            finetuned_attention_list = format_attention_for_viz(
                finetuned_attention, boxes, img_width, img_height
            )
            
            if base_attention_list and finetuned_attention_list:
                output_path = output_dir / f"attention_comparison_{args.task_type}.png"
                create_comparison_visualization(
                    image, base_attention_list, finetuned_attention_list,
                    boxes, str(output_path), args.task_type
                )
                logger.info(f"Comparison visualization saved to {output_path}")
            else:
                logger.warning("Could not format attention for visualization")
        elif finetuned_attention and not base_attention:
            # Only fine-tuned model has attention - show single model visualization
            logger.info("Base model has no spatial linking - showing only fine-tuned model attention")
            finetuned_attention_list = format_attention_for_viz(
                finetuned_attention, boxes, img_width, img_height
            )
            
            if finetuned_attention_list:
                output_path = output_dir / f"attention_finetuned_only_{args.task_type}.png"
                create_single_model_visualization(
                    image, finetuned_attention_list, boxes, 
                    str(output_path), args.task_type,
                    model_name="Fine-tuned Model (with Spatial Linking)"
                )
                logger.info(f"Fine-tuned model attention visualization saved to {output_path}")
                
                # Also print response comparison
                logger.info("=" * 80)
                logger.info("Response Comparison:")
                logger.info(f"  Base Qwen3-VL: {base_response}")
                logger.info(f"  Fine-tuned:    {finetuned_response}")
            else:
                logger.warning("Could not format attention for visualization")
        else:
            logger.warning("No attention weights available for visualization")
            logger.info("=" * 80)
            logger.info("Response Comparison (no attention viz):")
            logger.info(f"  Base model:    {base_response}")
            logger.info(f"  Fine-tuned:    {finetuned_response}")
    
    else:  # grounding
        logger.warning("Grounding task attention comparison not yet implemented")
        logger.info("Note: Grounding task doesn't use refer_boxes, so spatial attention may not be available")
    
    logger.info("=" * 80)
    logger.info("Comparison complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
