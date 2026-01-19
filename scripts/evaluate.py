#!/usr/bin/env python3
"""
Evaluation Script for Spatial Linking HOI Model.

Evaluates trained models on HOI grounding and referring tasks
using metrics from hoi-benchmarks. Includes attention visualization.

Usage:
    # Evaluate referring task
    python scripts/evaluate.py \
        --model_path outputs/spatial_linking_sft_dgx/run_XXXX \
        --test_file /dataset/benchmarks_simplified/hico_action_referring_test_simplified.json \
        --task_type referring

    # Evaluate grounding task
    python scripts/evaluate.py \
        --model_path outputs/spatial_linking_sft_dgx/run_XXXX \
        --test_file /dataset/benchmarks_simplified/hico_ground_test_simplified.json \
        --task_type grounding

    # With attention visualization
    python scripts/evaluate.py \
        --model_path outputs/spatial_linking_sft_dgx/run_XXXX \
        --test_file /dataset/benchmarks_simplified/hico_action_referring_test_simplified.json \
        --task_type referring \
        --visualize_attention \
        --viz_output_dir ./attention_viz
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import torch
from transformers import AutoProcessor
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.spatial_model import SpatialLinkingInteractionModel, compute_interaction_box

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SYSTEM PROMPT (same as training)
# =============================================================================

SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name":"zoom_in","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox_2d). Coordinates use 1000x1000 normalized format.","parameters":{"properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2] in 1000x1000 normalized format, where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"target_image":{"type":"number","description":"The index of the image to zoom in on. Use 1 for the main image."}},"required":["bbox_2d", "target_image"], "type":"object"},"args_format": "Format the arguments as a JSON object."}}
</tools>

For the function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model(model_path: str, device: str = "cuda"):
    """Load trained model and processor."""
    logger.info(f"Loading model from {model_path}")
    
    # Try to load processor from model path, fallback to base model
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
    except:
        logger.info("Loading processor from base model")
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            trust_remote_code=True,
        )
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    model = SpatialLinkingInteractionModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    
    # Set box token IDs
    model.set_box_token_ids(processor.tokenizer)
    model.eval()
    
    return model, processor


def load_test_data(test_file: str) -> List[Dict]:
    """Load test data from JSON file."""
    with open(test_file, 'r') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} test samples")
    return data


def get_image_path(file_name: str, image_base_dir: str) -> str:
    """Get full image path."""
    # Try direct path first
    direct_path = Path(image_base_dir) / file_name
    if direct_path.exists():
        return str(direct_path)
    
    # HICO-DET paths
    if "HICO" in file_name or "hico" in str(image_base_dir).lower():
        for subdir in ["train2015", "test2015"]:
            path = Path(image_base_dir) / "hico_20160224_det" / "images" / subdir / file_name
            if path.exists():
                return str(path)
    
    # SWIG paths
    swig_path = Path(image_base_dir) / "swig_hoi" / "images" / file_name
    if swig_path.exists():
        return str(swig_path)
    
    return str(direct_path)


# =============================================================================
# PROMPT CONSTRUCTION (matching training format)
# =============================================================================

def create_referring_prompt(person_box: List[int], object_box: List[int]) -> str:
    """Create referring task prompt matching training format."""
    return (
        f"<image> Question: What action is the person performing with the object?\n"
        f"The person is located at {person_box} and the object is at {object_box}.\n"
        f"Respond with ONLY the action phrase in format: \"{{verb}} {{object}}\" "
        f"(e.g., \"riding bicycle\", \"holding cup\"). Use base verb form, no articles.\n"
        f"Think in the mind first, and then decide whether to call tools one or more times "
        f"OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> "
        f"<tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."
    )


def create_grounding_prompt(action: str, object_category: str) -> str:
    """Create grounding task prompt matching training format."""
    return (
        f"<image> Question: Locate every person who is {action} {object_category} "
        f"and the {object_category} they interact with.\n"
        f"For each person-object pair, output bbox coordinates in JSON format like: "
        f"{{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"description\"}}. "
        f"Coordinates should be in 1000x1000 normalized format.\n"
        f"Think in the mind first, and then decide whether to call tools one or more times "
        f"OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> "
        f"<tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."
    )


def parse_answer(generated_text: str) -> str:
    """Parse answer from generated text with <think>...<answer> format."""
    import re
    
    # Try to extract from <answer> tags
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', generated_text, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()
    
    # Try to extract after last </think>
    if '</think>' in generated_text:
        parts = generated_text.split('</think>')
        return parts[-1].strip()
    
    # Fallback to last line
    lines = generated_text.strip().split('\n')
    return lines[-1].strip() if lines else generated_text


def parse_box_predictions(text: str) -> List[Dict]:
    """Parse box predictions from generated text."""
    import re
    
    pairs = []
    
    # Extract answer content first
    answer = parse_answer(text)
    
    # Try to find JSON array
    try:
        match = re.search(r'\[.*\]', answer, re.DOTALL)
        if match:
            boxes_data = json.loads(match.group())
            
            # Parse pairs
            for i in range(0, len(boxes_data), 2):
                if i + 1 < len(boxes_data):
                    person = boxes_data[i]
                    obj = boxes_data[i + 1]
                    
                    pairs.append({
                        "person_box": person.get("bbox_2d", [0, 0, 0, 0]),
                        "object_box": obj.get("bbox_2d", [0, 0, 0, 0]),
                    })
    except:
        pass
    
    return pairs


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_referring_task(
    model,
    processor,
    test_data: List[Dict],
    image_base_dir: str,
    max_samples: Optional[int] = None,
    visualize_attention: bool = False,
    viz_output_dir: Optional[str] = None,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate on referring task (action prediction)."""
    
    from utils.metrics import evaluate_referring, evaluate_referring_simple
    
    predictions = []
    references = []
    detailed_results = []
    
    samples = test_data[:max_samples] if max_samples else test_data
    device = next(model.parameters()).device
    
    # Setup visualization if needed
    if visualize_attention and viz_output_dir:
        Path(viz_output_dir).mkdir(parents=True, exist_ok=True)
        try:
            from utils.visualization import visualize_multi_region_attention
            has_viz = True
        except ImportError:
            logger.warning("Visualization not available, skipping attention viz")
            has_viz = False
    else:
        has_viz = False
    
    for idx, sample in enumerate(tqdm(samples, desc="Evaluating referring")):
        try:
            # Get data
            file_name = sample["file_name"]
            boxes = sample.get("boxes_1000", sample.get("boxes", []))
            person_idx = sample.get("person_box_idx", 0)
            object_idx = sample.get("object_box_idx", 1)
            gt_action = sample.get("gt_action", sample.get("response", ""))
            
            if len(boxes) < 2:
                continue
            
            person_box = boxes[person_idx]
            object_box = boxes[object_idx]
            interaction_box = [
                min(person_box[0], object_box[0]),
                min(person_box[1], object_box[1]),
                max(person_box[2], object_box[2]),
                max(person_box[3], object_box[3])
            ]
            
            # Load image
            image_path = get_image_path(file_name, image_base_dir)
            if not Path(image_path).exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            image = Image.open(image_path).convert("RGB")
            
            # Create prompt (matching training format)
            user_prompt = create_referring_prompt(person_box, object_box)
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            text = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs
            inputs = processor(text=text, images=[image], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Prepare refer_boxes for spatial linking
            refer_boxes_tensor = torch.tensor(
                [person_box, object_box, interaction_box], 
                dtype=torch.float32
            ).unsqueeze(0)  # [1, 3, 4]
            
            # Generate with spatial attention output if visualizing
            with torch.no_grad():
                # For visualization, we need to do a forward pass first
                if has_viz and idx < 10:  # Only visualize first 10
                    # Forward pass to get attention weights
                    _ = model(
                        **inputs,
                        refer_boxes=[refer_boxes_tensor.squeeze(0).to(device)],
                        output_spatial_attentions=True,
                    )
                    attention_info = model.get_spatial_attention_weights()
                    
                    if attention_info and attention_info[0]:
                        viz_path = Path(viz_output_dir) / f"attention_{idx:04d}.png"
                        try:
                            visualize_multi_region_attention(
                                image=image,
                                attention_info_list=attention_info[0],
                                output_path=str(viz_path),
                            )
                        except Exception as e:
                            logger.warning(f"Visualization failed: {e}")
                
                # Generate response
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            
            # Decode
            generated = processor.decode(outputs[0], skip_special_tokens=True)
            pred = parse_answer(generated)
            
            predictions.append(pred)
            references.append(gt_action)
            
            detailed_results.append({
                "file_name": file_name,
                "person_box": person_box,
                "object_box": object_box,
                "gt_action": gt_action,
                "predicted": pred,
                "full_response": generated,
            })
            
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            continue
    
    # Compute metrics
    logger.info(f"Computing BERTScore for {len(predictions)} samples...")
    
    try:
        metrics = evaluate_referring(predictions, references)
    except Exception as e:
        logger.warning(f"BERTScore failed: {e}, using simple metrics")
        metrics = evaluate_referring_simple(predictions, references)
    
    metrics["num_samples"] = len(predictions)
    
    return metrics, detailed_results


def evaluate_grounding_task(
    model,
    processor,
    test_data: List[Dict],
    image_base_dir: str,
    max_samples: Optional[int] = None,
    visualize_attention: bool = False,
    viz_output_dir: Optional[str] = None,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate on grounding task (box prediction)."""
    
    from utils.metrics import evaluate_grounding
    
    results = []
    detailed_results = []
    
    samples = test_data[:max_samples] if max_samples else test_data
    device = next(model.parameters()).device
    
    for idx, sample in enumerate(tqdm(samples, desc="Evaluating grounding")):
        try:
            file_name = sample["file_name"]
            action = sample["action"]
            object_category = sample["object_category"]
            boxes = sample.get("boxes_1000", sample.get("boxes", []))
            num_pairs = sample.get("num_pairs", 1)
            
            # Build ground truth pairs
            gt_pairs = []
            for i in range(num_pairs):
                if i * 2 + 1 < len(boxes):
                    gt_pairs.append({
                        "person_box": boxes[i * 2],
                        "object_box": boxes[i * 2 + 1]
                    })
            
            if not gt_pairs:
                continue
            
            # Load image
            image_path = get_image_path(file_name, image_base_dir)
            if not Path(image_path).exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            image = Image.open(image_path).convert("RGB")
            
            # Create prompt (matching training format)
            user_prompt = create_grounding_prompt(action, object_category)
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            text = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            inputs = processor(text=text, images=[image], return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            
            generated = processor.decode(outputs[0], skip_special_tokens=True)
            pred_pairs = parse_box_predictions(generated)
            
            results.append({
                "predicted_pairs": pred_pairs,
                "gt_pairs": gt_pairs,
            })
            
            detailed_results.append({
                "file_name": file_name,
                "action": action,
                "object_category": object_category,
                "gt_pairs": gt_pairs,
                "predicted_pairs": pred_pairs,
                "full_response": generated,
            })
            
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            continue
    
    # Compute metrics
    logger.info(f"Computing grounding metrics for {len(results)} samples...")
    metrics = evaluate_grounding(
        [r["predicted_pairs"] for r in results],
        [r["gt_pairs"] for r in results],
    )
    
    metrics["num_samples"] = len(results)
    
    return metrics, detailed_results


def print_metrics(metrics: Dict, title: str = "Evaluation Results"):
    """Print metrics in a nice format."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)
    
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    print("=" * 60 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate Spatial Linking HOI Model")
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--task_type",
        type=str,
        choices=["referring", "grounding"],
        required=True,
        help="Task type to evaluate"
    )
    parser.add_argument(
        "--image_base_dir",
        type=str,
        default="/workspace/dataset",
        help="Base directory for images"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--visualize_attention",
        action="store_true",
        help="Generate attention visualizations"
    )
    parser.add_argument(
        "--viz_output_dir",
        type=str,
        default="./attention_viz",
        help="Output directory for attention visualizations"
    )
    parser.add_argument(
        "--save_detailed",
        action="store_true",
        help="Save detailed per-sample results"
    )
    
    args = parser.parse_args()
    
    # Load model
    model, processor = load_model(args.model_path)
    
    # Load test data
    test_data = load_test_data(args.test_file)
    
    # Evaluate
    if args.task_type == "referring":
        metrics, detailed_results = evaluate_referring_task(
            model, processor, test_data,
            args.image_base_dir, args.max_samples,
            args.visualize_attention, args.viz_output_dir
        )
    else:
        metrics, detailed_results = evaluate_grounding_task(
            model, processor, test_data,
            args.image_base_dir, args.max_samples,
            args.visualize_attention, args.viz_output_dir
        )
    
    # Print results
    print_metrics(metrics, title=f"{args.task_type.upper()} Evaluation Results")
    
    # Save results
    if args.output_file:
        output_data = {"metrics": metrics}
        if args.save_detailed:
            output_data["detailed_results"] = detailed_results
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Results saved to {args.output_file}")
    
    # Log attention viz location
    if args.visualize_attention:
        logger.info(f"Attention visualizations saved to {args.viz_output_dir}")


if __name__ == "__main__":
    main()
