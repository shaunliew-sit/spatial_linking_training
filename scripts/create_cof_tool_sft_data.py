#!/usr/bin/env python3
"""
Create Chain-of-Focus (CoF) Tool-Calling SFT dataset.

This script generates multi-turn SFT data for training tool-augmented reasoning
on HOI tasks. Following the quality-over-quantity approach with ~5K samples.

Key Features:
- Multi-turn conversations with <think>, <tool_call>, <answer> format
- zoom_in tool for visual exploration
- Mix of tool-use (30-40%) and direct-answer (60-70%) samples
- Compatible with verl-tool framework

Reference:
- Format from dataset/hoi_cof_sft/hoi_cof_sft_data.json
- Script pattern from verl-tool/scripts/create_hoi_sft_dataset.py

Usage:
    python scripts/create_cof_tool_sft_data.py \
        --input-dir /dataset/benchmarks_simplified \
        --output-file /dataset/cof_tool_sft_data.json \
        --num-samples 5000 \
        --tool-ratio 0.35
"""

import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

ZOOM_IN_TOOL = {
    "type": "function",
    "function": {
        "name": "zoom_in",
        "description": "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox_2d). Coordinates use 1000x1000 normalized format.",
        "parameters": {
            "properties": {
                "bbox_2d": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                    "description": "The bounding box of the region to zoom in, as [x1, y1, x2, y2] in 1000x1000 normalized format."
                },
                "target_image": {
                    "type": "number",
                    "description": "The index of the image to zoom in on. Use 1 for the main image."
                }
            },
            "required": ["bbox_2d", "target_image"],
            "type": "object"
        },
        "args_format": "Format the arguments as a JSON object."
    }
}


def create_system_prompt() -> str:
    """Create system prompt with tool definitions."""
    tool_json = json.dumps(ZOOM_IN_TOOL)
    
    return f"""You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_json}
</tools>

For the function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


SYSTEM_PROMPT = create_system_prompt()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_interaction_box(person_box: List[int], object_box: List[int], padding: int = 50) -> List[int]:
    """Compute padded union box for zooming."""
    x1 = max(0, min(person_box[0], object_box[0]) - padding)
    y1 = max(0, min(person_box[1], object_box[1]) - padding)
    x2 = min(1000, max(person_box[2], object_box[2]) + padding)
    y2 = min(1000, max(person_box[3], object_box[3]) + padding)
    return [x1, y1, x2, y2]


def create_tool_call(bbox: List[int]) -> str:
    """Create a tool call string."""
    args = {"bbox_2d": bbox, "target_image": 1}
    return f'<tool_call>\n{{"name": "zoom_in", "arguments": {json.dumps(args)}}}\n</tool_call>'


def get_image_path(file_name: str, dataset_type: str, image_base_dir: str) -> str:
    """Get the full image path based on dataset type."""
    if dataset_type == "hico":
        if "train" in file_name:
            return f"{image_base_dir}/hico_20160224_det/images/train2015/{file_name}"
        else:
            return f"{image_base_dir}/hico_20160224_det/images/test2015/{file_name}"
    else:
        return f"{image_base_dir}/swig_hoi/images/{file_name}"


# =============================================================================
# REFERRING TASK TEMPLATES
# =============================================================================

def create_referring_with_tool(
    file_name: str,
    person_box: List[int],
    object_box: List[int],
    object_category: str,
    action: str,
    dataset_type: str,
    image_base_dir: str,
) -> Dict:
    """Create referring sample WITH tool usage (zoom_in)."""
    
    interaction_box = compute_interaction_box(person_box, object_box)
    image_path = get_image_path(file_name, dataset_type, image_base_dir)
    
    # User query
    user_query = (
        f"<image> Question: What action is the person performing with the object?\n"
        f"The person is located at {person_box} and the object is at {object_box}.\n"
        f"Respond with ONLY the action phrase in format: \"{{verb}} {{object}}\" "
        f"(e.g., \"riding bicycle\", \"holding cup\"). Use base verb form, no articles.\n"
        f"Think in the mind first, and then decide whether to call tools one or more times "
        f"OR provide final answer. Format strictly as: <think>...</think> "
        f"<tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) "
        f"OR <answer>...</answer> (if no tools needed)."
    )
    
    # First assistant turn - reasoning + tool call
    think_1 = (
        f"To identify the action between the person and object, I need to examine the "
        f"interaction region more closely. The person is at {person_box} and the object "
        f"is at {object_box}. I will zoom in on the interaction area to see the details "
        f"of how they are interacting."
    )
    tool_call = create_tool_call(interaction_box)
    assistant_1 = f"<think> {think_1} </think>\n{tool_call}"
    
    # Tool result - second user turn
    user_2 = (
        f"<image>\n"
        f"Think in the mind first, and then decide whether to call tools one or more times "
        f"OR provide final answer. Format strictly as: <think>...</think> "
        f"<tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) "
        f"OR <answer>...</answer> (if no tools needed)."
    )
    
    # Second assistant turn - final answer
    think_2 = (
        f"Now I can clearly see the interaction between the person and the {object_category}. "
        f"Based on the person's posture and position relative to the {object_category}, "
        f"the action is {action.split()[0] if ' ' in action else action}."
    )
    assistant_2 = f"<think> {think_2} </think>\n<answer> {action} </answer>"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": assistant_1},
        {"role": "user", "content": user_2},
        {"role": "assistant", "content": assistant_2}
    ]
    
    # For training, we'd need both original and cropped images
    # Here we just reference the original (cropped would be generated during training)
    return {
        "messages": messages,
        "images": [image_path, f"{image_path.rsplit('.', 1)[0]}_crop.jpg"],
        "task_type": "referring",
        "uses_tool": True,
        "metadata": {
            "file_name": file_name,
            "action": action,
            "object_category": object_category,
            "person_box": person_box,
            "object_box": object_box,
            "dataset": dataset_type
        }
    }


def create_referring_direct(
    file_name: str,
    person_box: List[int],
    object_box: List[int],
    object_category: str,
    action: str,
    dataset_type: str,
    image_base_dir: str,
) -> Dict:
    """Create referring sample WITHOUT tool usage (direct answer)."""
    
    image_path = get_image_path(file_name, dataset_type, image_base_dir)
    
    # User query
    user_query = (
        f"<image> Question: What action is the person performing with the object?\n"
        f"The person is located at {person_box} and the object is at {object_box}.\n"
        f"Respond with ONLY the action phrase in format: \"{{verb}} {{object}}\" "
        f"(e.g., \"riding bicycle\", \"holding cup\"). Use base verb form, no articles.\n"
        f"Think in the mind first, and then decide whether to call tools one or more times "
        f"OR provide final answer. Format strictly as: <think>...</think> "
        f"<tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) "
        f"OR <answer>...</answer> (if no tools needed)."
    )
    
    # Direct answer
    think = (
        f"The interaction region is clearly visible. The person is at {person_box} "
        f"and the {object_category} is at {object_box}. Based on the person's posture "
        f"and position, they are {action.split()[0] if ' ' in action else action} "
        f"the {object_category}."
    )
    assistant = f"<think> {think} </think>\n<answer> {action} </answer>"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": assistant}
    ]
    
    return {
        "messages": messages,
        "images": [image_path],
        "task_type": "referring",
        "uses_tool": False,
        "metadata": {
            "file_name": file_name,
            "action": action,
            "object_category": object_category,
            "person_box": person_box,
            "object_box": object_box,
            "dataset": dataset_type
        }
    }


# =============================================================================
# GROUNDING TASK TEMPLATES
# =============================================================================

def create_grounding_with_tool(
    file_name: str,
    person_box: List[int],
    object_box: List[int],
    object_category: str,
    action: str,
    dataset_type: str,
    image_base_dir: str,
) -> Dict:
    """Create grounding sample WITH tool usage."""
    
    interaction_box = compute_interaction_box(person_box, object_box)
    image_path = get_image_path(file_name, dataset_type, image_base_dir)
    
    user_query = (
        f"<image> Question: Locate every person who is {action} {object_category} "
        f"and the {object_category} they interact with.\n"
        f"For each person-object pair, output bbox coordinates in JSON format like: "
        f"{{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"description\"}}. "
        f"Coordinates should be in 1000x1000 normalized format.\n"
        f"Think in the mind first, and then decide whether to call tools one or more times "
        f"OR provide final answer."
    )
    
    # First turn - scan image
    think_1 = (
        f"I need to find all person-{object_category} pairs where the person is {action} "
        f"the {object_category}. Let me zoom in on potential interaction areas to examine them closely."
    )
    tool_call = create_tool_call(interaction_box)
    assistant_1 = f"<think> {think_1} </think>\n{tool_call}"
    
    user_2 = "<image>\nProvide final answer based on your observation."
    
    # Final answer
    response_pairs = [
        {"bbox_2d": person_box, "label": "person"},
        {"bbox_2d": object_box, "label": object_category}
    ]
    think_2 = (
        f"I can clearly identify the person-{object_category} pair. "
        f"The person is {action} the {object_category}."
    )
    assistant_2 = f"<think> {think_2} </think>\n<answer> {json.dumps(response_pairs)} </answer>"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": assistant_1},
        {"role": "user", "content": user_2},
        {"role": "assistant", "content": assistant_2}
    ]
    
    return {
        "messages": messages,
        "images": [image_path, f"{image_path.rsplit('.', 1)[0]}_crop.jpg"],
        "task_type": "grounding",
        "uses_tool": True,
        "metadata": {
            "file_name": file_name,
            "action": action,
            "object_category": object_category,
            "person_box": person_box,
            "object_box": object_box,
            "dataset": dataset_type
        }
    }


def create_grounding_direct(
    file_name: str,
    person_box: List[int],
    object_box: List[int],
    object_category: str,
    action: str,
    dataset_type: str,
    image_base_dir: str,
) -> Dict:
    """Create grounding sample WITHOUT tool usage."""
    
    image_path = get_image_path(file_name, dataset_type, image_base_dir)
    
    user_query = (
        f"<image> Question: Locate every person who is {action} {object_category} "
        f"and the {object_category} they interact with.\n"
        f"For each person-object pair, output bbox coordinates in JSON format like: "
        f"{{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"description\"}}. "
        f"Coordinates should be in 1000x1000 normalized format.\n"
        f"Think in the mind first, and then provide final answer."
    )
    
    response_pairs = [
        {"bbox_2d": person_box, "label": "person"},
        {"bbox_2d": object_box, "label": object_category}
    ]
    think = (
        f"The person and the {object_category} are clearly visible in the image. "
        f"The person is {action} the {object_category}."
    )
    assistant = f"<think> {think} </think>\n<answer> {json.dumps(response_pairs)} </answer>"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": assistant}
    ]
    
    return {
        "messages": messages,
        "images": [image_path],
        "task_type": "grounding",
        "uses_tool": False,
        "metadata": {
            "file_name": file_name,
            "action": action,
            "object_category": object_category,
            "person_box": person_box,
            "object_box": object_box,
            "dataset": dataset_type
        }
    }


# =============================================================================
# DATASET PROCESSING
# =============================================================================

def process_referring_data(
    data: List[Dict],
    dataset_type: str,
    image_base_dir: str,
    tool_ratio: float = 0.35,
) -> Tuple[List[Dict], List[Dict]]:
    """Process referring data into tool and direct samples."""
    tool_samples = []
    direct_samples = []
    
    for item in data:
        boxes_1000 = item.get("boxes_1000", [])
        if len(boxes_1000) < 2:
            continue
        
        person_box = boxes_1000[item["person_box_idx"]]
        object_box = boxes_1000[item["object_box_idx"]]
        
        # Extract object category from response (last word typically)
        response = item["response"]
        parts = response.split()
        object_category = parts[-1] if len(parts) > 1 else "object"
        action = response
        
        # Randomly assign to tool or direct
        if random.random() < tool_ratio:
            sample = create_referring_with_tool(
                item["file_name"], person_box, object_box,
                object_category, action, dataset_type, image_base_dir
            )
            tool_samples.append(sample)
        else:
            sample = create_referring_direct(
                item["file_name"], person_box, object_box,
                object_category, action, dataset_type, image_base_dir
            )
            direct_samples.append(sample)
    
    return tool_samples, direct_samples


def process_grounding_data(
    data: List[Dict],
    dataset_type: str,
    image_base_dir: str,
    tool_ratio: float = 0.35,
) -> Tuple[List[Dict], List[Dict]]:
    """Process grounding data into tool and direct samples."""
    tool_samples = []
    direct_samples = []
    
    for item in data:
        boxes_1000 = item.get("boxes_1000", [])
        if len(boxes_1000) < 2:
            continue
        
        person_box = boxes_1000[0]
        object_box = boxes_1000[1]
        action = item["action"]
        object_category = item["object_category"]
        
        if random.random() < tool_ratio:
            sample = create_grounding_with_tool(
                item["file_name"], person_box, object_box,
                object_category, action, dataset_type, image_base_dir
            )
            tool_samples.append(sample)
        else:
            sample = create_grounding_direct(
                item["file_name"], person_box, object_box,
                object_category, action, dataset_type, image_base_dir
            )
            direct_samples.append(sample)
    
    return tool_samples, direct_samples


def create_cof_sft_dataset(
    input_dir: str,
    output_file: str,
    image_base_dir: str,
    num_samples: int = 5000,
    tool_ratio: float = 0.35,
    seed: int = 42,
) -> Dict:
    """Create the CoF tool-calling SFT dataset."""
    random.seed(seed)
    input_path = Path(input_dir)
    
    all_tool_samples = []
    all_direct_samples = []
    
    stats = defaultdict(int)
    
    # Process HICO referring
    hico_ref_file = input_path / "hico_referring_train_simplified.json"
    if hico_ref_file.exists():
        logger.info(f"Processing HICO referring: {hico_ref_file}")
        with open(hico_ref_file) as f:
            data = json.load(f)
        tool, direct = process_referring_data(data, "hico", image_base_dir, tool_ratio)
        all_tool_samples.extend(tool)
        all_direct_samples.extend(direct)
        stats["hico_referring_tool"] = len(tool)
        stats["hico_referring_direct"] = len(direct)
    
    # Process HICO grounding
    hico_ground_file = input_path / "hico_ground_train_simplified.json"
    if hico_ground_file.exists():
        logger.info(f"Processing HICO grounding: {hico_ground_file}")
        with open(hico_ground_file) as f:
            data = json.load(f)
        tool, direct = process_grounding_data(data, "hico", image_base_dir, tool_ratio)
        all_tool_samples.extend(tool)
        all_direct_samples.extend(direct)
        stats["hico_grounding_tool"] = len(tool)
        stats["hico_grounding_direct"] = len(direct)
    
    # Process SWIG referring
    swig_ref_file = input_path / "swig_referring_train_simplified.json"
    if swig_ref_file.exists():
        logger.info(f"Processing SWIG referring: {swig_ref_file}")
        with open(swig_ref_file) as f:
            data = json.load(f)
        tool, direct = process_referring_data(data, "swig", image_base_dir, tool_ratio)
        all_tool_samples.extend(tool)
        all_direct_samples.extend(direct)
        stats["swig_referring_tool"] = len(tool)
        stats["swig_referring_direct"] = len(direct)
    
    # Process SWIG grounding
    swig_ground_file = input_path / "swig_ground_train_simplified.json"
    if swig_ground_file.exists():
        logger.info(f"Processing SWIG grounding: {swig_ground_file}")
        with open(swig_ground_file) as f:
            data = json.load(f)
        tool, direct = process_grounding_data(data, "swig", image_base_dir, tool_ratio)
        all_tool_samples.extend(tool)
        all_direct_samples.extend(direct)
        stats["swig_grounding_tool"] = len(tool)
        stats["swig_grounding_direct"] = len(direct)
    
    logger.info(f"Total tool samples: {len(all_tool_samples)}")
    logger.info(f"Total direct samples: {len(all_direct_samples)}")
    
    # Calculate target split
    target_tool = int(num_samples * tool_ratio)
    target_direct = num_samples - target_tool
    
    # Sample
    if len(all_tool_samples) > target_tool:
        all_tool_samples = random.sample(all_tool_samples, target_tool)
    if len(all_direct_samples) > target_direct:
        all_direct_samples = random.sample(all_direct_samples, target_direct)
    
    final_samples = all_tool_samples + all_direct_samples
    random.shuffle(final_samples)
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(final_samples, f, indent=2)
    
    stats["total"] = len(final_samples)
    stats["tool_final"] = len(all_tool_samples)
    stats["direct_final"] = len(all_direct_samples)
    
    logger.info(f"Final dataset: {len(final_samples)} samples")
    logger.info(f"  With tools: {len(all_tool_samples)} ({len(all_tool_samples)/len(final_samples)*100:.1f}%)")
    logger.info(f"  Direct: {len(all_direct_samples)} ({len(all_direct_samples)/len(final_samples)*100:.1f}%)")
    logger.info(f"Saved to: {output_file}")
    
    # Save stats
    stats_file = output_path.with_suffix('.stats.json')
    with open(stats_file, 'w') as f:
        json.dump(dict(stats), f, indent=2)
    
    return dict(stats)


def main():
    parser = argparse.ArgumentParser(
        description="Create Chain-of-Focus tool-calling SFT dataset"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/dataset/benchmarks_simplified",
        help="Directory containing simplified benchmark files"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="/dataset/cof_tool_sft_data.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--image-base-dir",
        type=str,
        default="/dataset",
        help="Base directory for images"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Target number of samples (default: 5000)"
    )
    parser.add_argument(
        "--tool-ratio",
        type=float,
        default=0.35,
        help="Ratio of samples using tools (default: 0.35)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    stats = create_cof_sft_dataset(
        input_dir=args.input_dir,
        output_file=args.output_file,
        image_base_dir=args.image_base_dir,
        num_samples=args.num_samples,
        tool_ratio=args.tool_ratio,
        seed=args.seed,
    )
    
    print("\n" + "=" * 60)
    print("COF TOOL SFT DATASET CREATION COMPLETE")
    print("=" * 60)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
