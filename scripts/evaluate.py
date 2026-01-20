#!/usr/bin/env python3
"""
Evaluation Script for Spatial Linking HOI Model.

Evaluates trained models on HOI grounding and referring tasks
using metrics from hoi-benchmarks:
- Referring: METEOR, CIDEr, BLEU, ROUGE-L (COCO caption metrics)
- Grounding: AR (Average Recall) at various IoU thresholds

Usage:
    # Evaluate referring task
    python scripts/evaluate.py \
        --model_path outputs/spatial_linking_sft_dgx/run_XXXX \
        --test_file /dataset/benchmarks_simplified/hico_action_referring_test_simplified.json \
        --task_type referring \
        --verbose

    # Evaluate grounding task
    python scripts/evaluate.py \
        --model_path outputs/spatial_linking_sft_dgx/run_XXXX \
        --test_file /dataset/benchmarks_simplified/hico_ground_test_simplified.json \
        --task_type grounding \
        --verbose

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
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from tqdm import tqdm

import torch
from transformers import AutoProcessor
from PIL import Image

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.spatial_model import SpatialLinkingInteractionModel, compute_interaction_box

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
    import warnings
    import os
    
    logger.info(f"Loading model from {model_path}")
    
    # Suppress known deprecation and compatibility warnings
    warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
    warnings.filterwarnings("ignore", message=".*Unexpected keyword arguments.*LoraConfig.*")
    warnings.filterwarnings("ignore", message=".*Some weights.*were not initialized.*")
    
    # Try to load processor from model path, fallback to base model
    try:
        processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
    except Exception:
        logger.info("Loading processor from base model")
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            trust_remote_code=True,
        )
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Load model with bfloat16 precision
    model = SpatialLinkingInteractionModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    
    # IMPORTANT: Load spatial linking weights if available
    # These are saved separately because PEFT only saves LoRA adapters
    spatial_linking_path = os.path.join(model_path, "spatial_linking.pt")
    if os.path.exists(spatial_linking_path):
        logger.info(f"Loading spatial linking weights from {spatial_linking_path}")
        spatial_linking_state = torch.load(spatial_linking_path, map_location="cpu")
        model.spatial_linking.load_state_dict(spatial_linking_state)
        logger.info(f"  Loaded keys: {list(spatial_linking_state.keys())}")
    else:
        logger.warning(f"Spatial linking weights not found at {spatial_linking_path}")
        logger.warning("  The spatial_linking module will use random initialization!")
        logger.warning("  This likely means the model was trained before the save fix was applied.")
    
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


def parse_response_components(generated_text: str) -> Dict[str, Any]:
    """
    Parse all components from generated text following the CoF format.
    
    Expected format from training:
        <think>reasoning...</think>
        <tool_call>{"name": "zoom_in", "arguments": {...}}</tool_call>
        OR
        <answer>final answer</answer>
    
    Returns:
        Dict with keys:
        - 'thinking': str - content of <think> tags
        - 'tool_calls': List[Dict] - parsed tool calls
        - 'answer': str - content of <answer> tags
        - 'has_tool_call': bool - whether tool_call was requested
        - 'raw_text': str - original text
    """
    result = {
        'thinking': '',
        'tool_calls': [],
        'answer': '',
        'has_tool_call': False,
        'raw_text': generated_text
    }
    
    # Extract thinking content (may appear multiple times in multi-turn)
    think_matches = re.findall(r'<think>(.*?)</think>', generated_text, re.DOTALL)
    if think_matches:
        result['thinking'] = ' '.join([t.strip() for t in think_matches])
    
    # Extract tool calls
    tool_call_matches = re.findall(r'<tool_call>(.*?)</tool_call>', generated_text, re.DOTALL)
    if tool_call_matches:
        result['has_tool_call'] = True
        for tc in tool_call_matches:
            try:
                # Parse JSON from tool call
                tc_clean = tc.strip()
                parsed = json.loads(tc_clean)
                result['tool_calls'].append(parsed)
            except json.JSONDecodeError:
                # Store as string if JSON parsing fails
                result['tool_calls'].append({'raw': tc_clean})
    
    # Extract answer content
    answer_match = re.search(r'<answer>\s*(.*?)\s*</answer>', generated_text, re.DOTALL)
    if answer_match:
        result['answer'] = answer_match.group(1).strip()
    
    return result


def parse_answer(generated_text: str) -> str:
    """Parse answer from generated text with <think>...<answer> format."""
    components = parse_response_components(generated_text)
    
    # If we have an explicit answer, return it
    if components['answer']:
        return components['answer']
    
    # If model requested tool_call, try to extract action hint from thinking
    if components['has_tool_call']:
        thinking = components['thinking']
        if thinking:
            # Look for action mentions in thinking like "the action is X"
            action_patterns = [
                r'the action (?:is|appears to be|seems to be)\s+(\w+(?:\s+\w+)?)',
                r'action is\s+(\w+ing)',
                r'they are\s+(\w+ing\s+\w+)',
            ]
            for pattern in action_patterns:
                action_match = re.search(pattern, thinking, re.IGNORECASE)
                if action_match:
                    return action_match.group(1).strip()
        # Tool call requested but no answer yet - this is expected behavior
        return "[TOOL_CALL_PENDING]"
    
    # Try to extract after last </think>
    if '</think>' in generated_text:
        parts = generated_text.split('</think>')
        after_think = parts[-1].strip()
        # Make sure it's not a tool call or empty
        if after_think and not after_think.startswith('<'):
            return after_think
    
    # Fallback: look for common action patterns in the text
    action_patterns = [
        r'(?:person is|they are|action is)\s+(\w+ing\s+\w+)',
        r'(\w+ing\s+(?:the\s+)?\w+)',
    ]
    for pattern in action_patterns:
        match = re.search(pattern, generated_text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    # Last resort: return empty
    return ""


def format_response_for_display(generated_text: str, max_len: int = 500) -> str:
    """Format response for verbose display, showing all components clearly."""
    components = parse_response_components(generated_text)
    
    lines = []
    
    # Show thinking (truncated if needed)
    if components['thinking']:
        thinking = components['thinking']
        if len(thinking) > max_len:
            thinking = thinking[:max_len] + "..."
        lines.append(f"  <think>: {thinking}")
    
    # Show tool calls
    if components['tool_calls']:
        for i, tc in enumerate(components['tool_calls']):
            if isinstance(tc, dict) and 'name' in tc:
                args = tc.get('arguments', {})
                lines.append(f"  <tool_call> [{i+1}]: {tc['name']}({json.dumps(args)})")
            else:
                lines.append(f"  <tool_call> [{i+1}]: {tc}")
    
    # Show answer
    if components['answer']:
        lines.append(f"  <answer>: {components['answer']}")
    elif components['has_tool_call']:
        lines.append(f"  <answer>: [Pending - model requested tool execution]")
    else:
        lines.append(f"  <answer>: [None extracted]")
    
    return '\n'.join(lines) if lines else "  [Empty response]"


def parse_box_predictions(text: str, img_shape: Tuple[int, int] = (1000, 1000)) -> List[Dict]:
    """
    Parse box predictions from generated text.
    
    Handles multiple JSON formats:
    - [{"person_bbox": [...], "object_bbox": [...]}, ...]
    - [{"bbox_2d": [...], "label": "person"}, {"bbox_2d": [...], "label": "object"}, ...]
    - [{"pair_id": 1, "person_bbox": [...], "object_bbox": [...]}, ...]
    
    Args:
        text: Generated text containing JSON
        img_shape: (height, width) for coordinate conversion
        
    Returns:
        List of dicts with 'person_box' and 'object_box' keys
    """
    pairs = []
    h, w = img_shape
    
    # Extract answer content first
    answer = parse_answer(text)
    
    # Clean markdown code fences if present
    cleaned_text = answer.strip()
    if cleaned_text.startswith("```"):
        lines = cleaned_text.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned_text = '\n'.join(lines)
    
    # Try to parse JSON
    try:
        # Find JSON array
        match = re.search(r'\[[\s\S]*\]', cleaned_text)
        if match:
            parsed = json.loads(match.group())
            
            if isinstance(parsed, list):
                # Check format type
                if len(parsed) > 0:
                    first_item = parsed[0]
                    
                    # Format 1: person_bbox/object_bbox pairs
                    if isinstance(first_item, dict) and ('person_bbox' in first_item or 'person_box' in first_item):
                        for det in parsed:
                            if not isinstance(det, dict):
                                continue
                            person_bbox = det.get('person_bbox', det.get('person_box', []))
                            object_bbox = det.get('object_bbox', det.get('object_box', []))
                            
                            if len(person_bbox) == 4 and len(object_bbox) == 4:
                                pairs.append({
                                    'person_box': person_bbox,
                                    'object_box': object_bbox
                                })
                    
                    # Format 2: bbox_2d with labels (alternating person/object)
                    elif isinstance(first_item, dict) and 'bbox_2d' in first_item:
                        for i in range(0, len(parsed) - 1, 2):
                            person_det = parsed[i]
                            object_det = parsed[i + 1] if i + 1 < len(parsed) else None
                            
                            if object_det is None:
                                continue
                            
                            person_bbox = person_det.get('bbox_2d', [])
                            object_bbox = object_det.get('bbox_2d', [])
                            
                            if len(person_bbox) == 4 and len(object_bbox) == 4:
                                pairs.append({
                                    'person_box': person_bbox,
                                    'object_box': object_bbox
                                })
    except (json.JSONDecodeError, Exception) as e:
        logger.debug(f"JSON parsing failed: {e}")
    
    return pairs


def clean_action_response(response_text: str) -> str:
    """Clean action response to extract action phrase."""
    response = response_text.strip()
    
    # Remove common prefixes
    prefixes_to_remove = [
        "the person is ",
        "person is ",
        "they are ",
        "action: ",
        "answer: ",
        "output: ",
    ]
    
    response_lower = response.lower()
    for prefix in prefixes_to_remove:
        if response_lower.startswith(prefix):
            response = response[len(prefix):].strip()
            response_lower = response.lower()
    
    # Remove trailing punctuation
    response = response.rstrip('.!?,;:')
    
    # Convert to lowercase
    response = response.lower().strip()
    
    return response


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
    verbose: bool = False,
) -> Tuple[Dict, List[Dict]]:
    """Evaluate on referring task (action prediction)."""
    
    from utils.metrics import evaluate_referring_coco, evaluate_referring_simple
    
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
    
    viz_count = 0  # Track saved visualizations
    
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
            img_width, img_height = image.size
            
            # Create prompt (matching training format)
            user_prompt = create_referring_prompt(person_box, object_box)
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            text = processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Process inputs with explicit truncation
            inputs = processor(
                text=text, 
                images=[image], 
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            )
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
                    try:
                        # Forward pass to get attention weights
                        _ = model(
                            **inputs,
                            refer_boxes=[refer_boxes_tensor.squeeze(0).to(device)],
                            output_spatial_attentions=True,
                        )
                        attention_info = model.get_spatial_attention_weights()
                        
                        if attention_info and len(attention_info) > 0:
                            # Format attention info for visualization
                            attention_info_list = format_attention_for_viz(
                                attention_info, 
                                [person_box, object_box, interaction_box],
                                img_width, img_height,
                                inputs.get('image_grid_thw')
                            )
                            
                            if attention_info_list:
                                viz_path = Path(viz_output_dir) / f"attention_{idx:04d}.png"
                                visualize_multi_region_attention(
                                    image=image,
                                    attention_info_list=attention_info_list,
                                    output_path=str(viz_path),
                                )
                                viz_count += 1
                    except Exception as e:
                        logger.debug(f"Visualization failed for sample {idx}: {e}")
                
                # Generate response
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            
            # Decode only the NEW tokens (not the full sequence)
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            generated = processor.decode(generated_ids, skip_special_tokens=True)
            
            answer = parse_answer(generated)
            pred = clean_action_response(answer)
            
            predictions.append(pred)
            references.append(gt_action)
            
            # Parse response components for detailed logging
            components = parse_response_components(generated)
            
            detailed_results.append({
                "file_name": file_name,
                "person_box": person_box,
                "object_box": object_box,
                "gt_action": gt_action,
                "predicted": pred,
                "raw_answer": answer,
                "full_response": generated,
                # Parsed components
                "thinking": components['thinking'],
                "tool_calls": components['tool_calls'],
                "has_tool_call": components['has_tool_call'],
                "answer_tag": components['answer'],
            })
            
            # Verbose output with full component breakdown
            if verbose and idx < 20:
                print(f"\n{'='*60}")
                print(f"[Sample {idx+1}] {file_name}")
                print(f"{'='*60}")
                print(f"  Person box: {person_box}")
                print(f"  Object box: {object_box}")
                print(f"  GT action: {gt_action}")
                print(f"\n  Model Response:")
                print(format_response_for_display(generated))
                print(f"\n  Final Prediction: {pred if pred else '[EMPTY]'}")
                
                # Show match status
                if pred and gt_action:
                    is_exact = pred.lower().strip() == gt_action.lower().strip()
                    is_partial = any(w in pred.lower() for w in gt_action.lower().split())
                    status = "✓ EXACT" if is_exact else ("~ PARTIAL" if is_partial else "✗ MISMATCH")
                    print(f"  Match Status: {status}")
            
        except Exception as e:
            logger.warning(f"Error processing sample {idx}: {e}")
            continue
    
    # Compute metrics
    logger.info(f"Computing COCO caption metrics for {len(predictions)} samples...")
    
    try:
        metrics = evaluate_referring_coco(predictions, references)
    except Exception as e:
        logger.warning(f"COCO evaluation failed: {e}, using simple metrics")
        metrics = evaluate_referring_simple(predictions, references)
    
    metrics["num_samples"] = len(predictions)
    
    if has_viz:
        logger.info(f"Saved {viz_count} attention visualizations to {viz_output_dir}")
    
    return metrics, detailed_results


def evaluate_grounding_task(
    model,
    processor,
    test_data: List[Dict],
    image_base_dir: str,
    max_samples: Optional[int] = None,
    visualize_attention: bool = False,
    viz_output_dir: Optional[str] = None,
    verbose: bool = False,
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
            gt_box_inds = sample.get("gt_box_inds", list(range(num_pairs * 2)))
            img_height = sample.get("height", 1000)
            img_width = sample.get("width", 1000)
            
            # Build ground truth pairs from simplified format
            gt_pairs = []
            for i in range(num_pairs):
                person_idx = gt_box_inds[i * 2]
                object_idx = gt_box_inds[i * 2 + 1]
                
                if person_idx < len(boxes) and object_idx < len(boxes):
                    gt_pairs.append({
                        "person_box": boxes[person_idx],
                        "object_box": boxes[object_idx]
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
            
            inputs = processor(
                text=text, 
                images=[image], 
                return_tensors="pt",
                truncation=True,
                max_length=4096,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=500,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            
            generated = processor.decode(outputs[0], skip_special_tokens=True)
            pred_pairs = parse_box_predictions(generated, (img_height, img_width))
            
            results.append({
                "predicted_pairs": pred_pairs,
                "gt_pairs": gt_pairs,
            })
            
            # Parse response components for detailed logging
            components = parse_response_components(generated)
            
            detailed_results.append({
                "file_name": file_name,
                "action": action,
                "object_category": object_category,
                "gt_pairs": gt_pairs,
                "predicted_pairs": pred_pairs,
                "num_pred": len(pred_pairs),
                "num_gt": len(gt_pairs),
                "full_response": generated,
                # Parsed components
                "thinking": components['thinking'],
                "tool_calls": components['tool_calls'],
                "has_tool_call": components['has_tool_call'],
                "answer_tag": components['answer'],
            })
            
            # Verbose output with full component breakdown
            if verbose and idx < 20:
                print(f"\n{'='*60}")
                print(f"[Sample {idx+1}] {file_name}")
                print(f"{'='*60}")
                print(f"  Action: {action} {object_category}")
                print(f"  GT pairs: {len(gt_pairs)}")
                
                print(f"\n  Model Response:")
                print(format_response_for_display(generated))
                
                print(f"\n  Predicted pairs: {len(pred_pairs)}")
                for i, pair in enumerate(pred_pairs[:3]):
                    print(f"    Pair {i+1}: Person {pair['person_box']}, Object {pair['object_box']}")
            
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


def format_attention_for_viz(
    attention_weights, 
    boxes: List[List[float]], 
    img_width: int, 
    img_height: int,
    image_grid_thw: Optional[torch.Tensor] = None
) -> List[Dict]:
    """
    Format attention weights for visualization.
    
    Args:
        attention_weights: Raw attention weights from model
        boxes: List of boxes [person, object, interaction]
        img_width: Image width
        img_height: Image height
        image_grid_thw: Grid dimensions from processor
        
    Returns:
        List of attention info dicts for visualization
    """
    if attention_weights is None or len(attention_weights) == 0:
        return []
    
    box_types = ['person', 'object', 'interaction']
    attention_info_list = []
    
    # Get grid dimensions
    if image_grid_thw is not None and image_grid_thw.numel() >= 3:
        grid_thw = image_grid_thw[0].tolist() if image_grid_thw.dim() > 1 else image_grid_thw.tolist()
    else:
        # Estimate grid size (Qwen3VL uses 14x14 patches)
        patch_size = 14
        grid_thw = [1, img_height // patch_size, img_width // patch_size]
    
    for i, (box, box_type) in enumerate(zip(boxes[:len(attention_weights)], box_types)):
        if i >= len(attention_weights):
            break
        
        attn = attention_weights[i]
        if attn is None:
            continue
        
        # Ensure attention is on CPU
        if torch.is_tensor(attn):
            attn = attn.detach().cpu()
        
        # Normalize box to [0, 1]
        norm_box = [
            box[0] / 1000.0,
            box[1] / 1000.0,
            box[2] / 1000.0,
            box[3] / 1000.0,
        ]
        
        # Create patch indices (dummy for now - actual implementation needs image_start_idx)
        num_patches = int(grid_thw[1] * grid_thw[2])
        patch_indices = torch.arange(num_patches)
        
        attention_info_list.append({
            'attention_weights': attn.unsqueeze(0) if attn.dim() == 3 else attn,
            'bbox': torch.tensor(norm_box),
            'box_type': box_type,
            'grid_thw': torch.tensor(grid_thw),
            'patch_indices': patch_indices,
        })
    
    return attention_info_list


def print_metrics(metrics: Dict, title: str = "Evaluation Results"):
    """Print metrics in a nice format."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)
    print(f"{'Metric':<20} {'Value':>10}  {'Description':<35}")
    print("-" * 70)
    
    # Define metric descriptions
    descriptions = {
        'METEOR': 'Semantic similarity',
        'CIDEr': 'Corpus consensus',
        'BLEU_1': 'Unigram overlap',
        'BLEU_2': 'Bigram overlap',
        'BLEU_3': 'Trigram overlap',
        'BLEU_4': '4-gram overlap',
        'ROUGE_L': 'Longest common subsequence',
        'exact_match': 'Exact string match',
        'verb_match': 'First word (verb) match',
        'word_overlap': 'Jaccard word similarity',
        'non_empty': 'Non-empty prediction rate',
        'AR': 'Average Recall @ IoU=0.50:0.95',
        'AR@0.5': 'Average Recall @ IoU=0.50',
        'AR@0.75': 'Average Recall @ IoU=0.75',
        'ARs': 'AR for small objects (area < 32^2)',
        'ARm': 'AR for medium objects',
        'ARl': 'AR for large objects (area > 96^2)',
        'num_samples': 'Total samples evaluated',
    }
    
    for key, value in metrics.items():
        desc = descriptions.get(key, '')
        if isinstance(value, float):
            if key in ['METEOR', 'CIDEr', 'BLEU_1', 'BLEU_2', 'BLEU_3', 'BLEU_4', 
                       'ROUGE_L', 'exact_match', 'verb_match', 'word_overlap', 'non_empty',
                       'AR', 'AR@0.5', 'AR@0.75', 'ARs', 'ARm', 'ARl']:
                print(f"{key:<20} {value*100:>9.2f}%  {desc:<35}")
            else:
                print(f"{key:<20} {value:>10.4f}  {desc:<35}")
        else:
            print(f"{key:<20} {value:>10}  {desc:<35}")
    
    print("=" * 70 + "\n")


def print_response_statistics(detailed_results: List[Dict]):
    """Print statistics about model response patterns (tool calls, answers, etc.)."""
    if not detailed_results:
        return
    
    total = len(detailed_results)
    has_thinking = sum(1 for r in detailed_results if r.get('thinking'))
    has_tool_call = sum(1 for r in detailed_results if r.get('has_tool_call'))
    has_answer = sum(1 for r in detailed_results if r.get('answer_tag'))
    empty_pred = sum(1 for r in detailed_results if not r.get('predicted'))
    
    print("=" * 70)
    print(" Response Pattern Statistics")
    print("=" * 70)
    print(f"  Total samples:           {total}")
    print(f"  Has <think> tag:         {has_thinking:>5} ({100*has_thinking/total:.1f}%)")
    print(f"  Has <tool_call> tag:     {has_tool_call:>5} ({100*has_tool_call/total:.1f}%)")
    print(f"  Has <answer> tag:        {has_answer:>5} ({100*has_answer/total:.1f}%)")
    print(f"  Empty predictions:       {empty_pred:>5} ({100*empty_pred/total:.1f}%)")
    print("-" * 70)
    
    # Tool call breakdown
    if has_tool_call > 0:
        tool_names = {}
        for r in detailed_results:
            for tc in r.get('tool_calls', []):
                if isinstance(tc, dict):
                    name = tc.get('name', 'unknown')
                    tool_names[name] = tool_names.get(name, 0) + 1
        
        if tool_names:
            print("  Tool calls breakdown:")
            for name, count in sorted(tool_names.items(), key=lambda x: -x[1]):
                print(f"    - {name}: {count}")
    
    print("=" * 70 + "\n")


def save_results(
    metrics: Dict,
    detailed_results: List[Dict],
    output_dir: str,
    task_type: str,
    save_detailed: bool = True,
):
    """Save evaluation results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save metrics
    metrics_file = output_path / f"{task_type}_metrics_{timestamp}.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_file}")
    
    # Save detailed results
    if save_detailed:
        detailed_file = output_path / f"{task_type}_detailed_{timestamp}.json"
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)
        logger.info(f"Detailed results saved to {detailed_file}")
    
    # Save summary log
    log_file = output_path / f"{task_type}_log_{timestamp}.txt"
    with open(log_file, 'w') as f:
        f.write(f"Evaluation Results - {task_type.upper()}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("METRICS:\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "=" * 60 + "\n")
        f.write("SAMPLE RESPONSES (first 10):\n\n")
        
        for i, result in enumerate(detailed_results[:10]):
            f.write(f"Sample {i+1}: {result.get('file_name', 'N/A')}\n")
            if 'gt_action' in result:
                f.write(f"  GT Action: {result['gt_action']}\n")
                f.write(f"  Predicted: {result['predicted']}\n")
            elif 'action' in result:
                f.write(f"  Action: {result['action']} {result.get('object_category', '')}\n")
                f.write(f"  GT pairs: {result.get('num_gt', len(result.get('gt_pairs', [])))}\n")
                f.write(f"  Pred pairs: {result.get('num_pred', len(result.get('predicted_pairs', [])))}\n")
            f.write(f"  Response: {result.get('full_response', '')[:200]}...\n\n")
    
    logger.info(f"Log saved to {log_file}")


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
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Output directory for results"
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
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed per-sample output during evaluation"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "=" * 70)
    print(" Spatial Linking HOI Model Evaluation")
    print("=" * 70)
    print(f"  Model:       {args.model_path}")
    print(f"  Test file:   {args.test_file}")
    print(f"  Task:        {args.task_type}")
    print(f"  Max samples: {args.max_samples or 'all'}")
    print(f"  Output dir:  {args.output_dir}")
    print(f"  Verbose:     {args.verbose}")
    print("=" * 70 + "\n")
    
    # Load model
    model, processor = load_model(args.model_path)
    
    # Load test data
    test_data = load_test_data(args.test_file)
    
    # Evaluate
    if args.task_type == "referring":
        metrics, detailed_results = evaluate_referring_task(
            model, processor, test_data,
            args.image_base_dir, args.max_samples,
            args.visualize_attention, args.viz_output_dir,
            args.verbose
        )
    else:
        metrics, detailed_results = evaluate_grounding_task(
            model, processor, test_data,
            args.image_base_dir, args.max_samples,
            args.visualize_attention, args.viz_output_dir,
            args.verbose
        )
    
    # Print results
    print_metrics(metrics, title=f"{args.task_type.upper()} Evaluation Results")
    
    # Print response pattern statistics
    print_response_statistics(detailed_results)
    
    # Save results
    save_results(
        metrics, detailed_results, 
        args.output_dir, args.task_type,
        args.save_detailed
    )
    
    # Log attention viz location
    if args.visualize_attention:
        logger.info(f"Attention visualizations saved to {args.viz_output_dir}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
