#!/usr/bin/env python3
"""
Create high-quality SFT dataset for Spatial Linking training.

This script generates SFT data for training the spatial linking module on HOI tasks
(referring + grounding) using HICO-DET and SWIG-HOI annotations.

Key Features:
- Quality filtering for clear, unambiguous interactions
- Balanced action distribution
- ShareGPT format compatible with TRL SFTTrainer
- Includes refer_boxes for spatial linking training

Usage:
    python scripts/create_spatial_sft_data.py \
        --input-dir /dataset/benchmarks_simplified \
        --output-file /dataset/spatial_sft_data.json \
        --num-samples 5000 \
        --image-base-dir /dataset
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
# CONSTANTS
# =============================================================================

# System prompt for spatial linking HOI tasks
SYSTEM_PROMPT = """You are an expert at analyzing human-object interactions in images. You can identify actions between people and objects based on their spatial positions and visual appearance."""

# Quality thresholds
MIN_BOX_AREA_RATIO = 0.001  # Minimum box area as ratio of image
MAX_BOX_AREA_RATIO = 0.9    # Maximum box area as ratio of image
MIN_BOX_SIZE = 10           # Minimum box dimension in 1000-grid


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compute_interaction_box(person_box: List[int], object_box: List[int]) -> List[int]:
    """Compute union box (interaction region) from person and object boxes."""
    return [
        min(person_box[0], object_box[0]),  # x1
        min(person_box[1], object_box[1]),  # y1
        max(person_box[2], object_box[2]),  # x2
        max(person_box[3], object_box[3])   # y2
    ]


def is_valid_box(box: List[int], threshold: int = MIN_BOX_SIZE) -> bool:
    """Check if a bounding box is valid (not too small)."""
    width = box[2] - box[0]
    height = box[3] - box[1]
    return width >= threshold and height >= threshold


def compute_box_area_ratio(box: List[int]) -> float:
    """Compute box area as ratio of 1000x1000 grid."""
    width = box[2] - box[0]
    height = box[3] - box[1]
    return (width * height) / (1000 * 1000)


def is_quality_sample(
    person_box: List[int], 
    object_box: List[int],
    min_area: float = MIN_BOX_AREA_RATIO,
    max_area: float = MAX_BOX_AREA_RATIO
) -> bool:
    """Check if a sample meets quality criteria."""
    # Check box validity
    if not is_valid_box(person_box) or not is_valid_box(object_box):
        return False
    
    # Check area ratios
    person_area = compute_box_area_ratio(person_box)
    object_area = compute_box_area_ratio(object_box)
    
    if person_area < min_area or person_area > max_area:
        return False
    if object_area < min_area or object_area > max_area:
        return False
    
    return True


def get_image_path(file_name: str, dataset_type: str, image_base_dir: str) -> str:
    """Get the full image path based on dataset type."""
    if dataset_type == "hico":
        # HICO images are in hico_20160224_det/images/train2015/ or test2015/
        if "train" in file_name:
            return f"{image_base_dir}/hico_20160224_det/images/train2015/{file_name}"
        else:
            return f"{image_base_dir}/hico_20160224_det/images/test2015/{file_name}"
    else:  # swig
        # SWIG images are in swig_hoi/images/
        return f"{image_base_dir}/swig_hoi/images/{file_name}"


# =============================================================================
# SAMPLE GENERATION
# =============================================================================

def create_referring_sample(
    sample: Dict,
    dataset_type: str,
    image_base_dir: str,
) -> Optional[Dict]:
    """
    Create a referring task sample (action prediction given boxes).
    
    Input: Person box + Object box → Output: Action description
    """
    file_name = sample["file_name"]
    boxes_1000 = sample["boxes_1000"]
    person_idx = sample["person_box_idx"]
    object_idx = sample["object_box_idx"]
    response = sample["response"]
    
    person_box = boxes_1000[person_idx]
    object_box = boxes_1000[object_idx]
    
    # Quality check
    if not is_quality_sample(person_box, object_box):
        return None
    
    # Compute interaction box
    interaction_box = compute_interaction_box(person_box, object_box)
    
    # Build conversation
    user_content = (
        f"<image> Question: What action is the person performing with the object?\n"
        f"The person is located at {person_box} and the object is at {object_box}.\n"
        f"Respond with ONLY the action phrase in format: \"{{verb}} {{object}}\" "
        f"(e.g., \"riding bicycle\", \"holding cup\"). Use base verb form, no articles."
    )
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response}
    ]
    
    image_path = get_image_path(file_name, dataset_type, image_base_dir)
    
    return {
        "messages": messages,
        "images": [image_path],
        "refer_boxes": [person_box, object_box, interaction_box],
        "task_type": "referring",
        "metadata": {
            "file_name": file_name,
            "action": response,
            "dataset": dataset_type
        }
    }


def create_grounding_sample(
    sample: Dict,
    dataset_type: str,
    image_base_dir: str,
) -> Optional[Dict]:
    """
    Create a grounding task sample (box prediction given action).
    
    Input: Action query → Output: Person + Object boxes
    """
    file_name = sample["file_name"]
    boxes_1000 = sample["boxes_1000"]
    action = sample["action"]
    object_category = sample["object_category"]
    num_pairs = sample["num_pairs"]
    
    # Get first pair for simplicity (can extend to multiple pairs)
    if len(boxes_1000) < 2:
        return None
    
    person_box = boxes_1000[0]
    object_box = boxes_1000[1]
    
    # Quality check
    if not is_quality_sample(person_box, object_box):
        return None
    
    # Compute interaction box
    interaction_box = compute_interaction_box(person_box, object_box)
    
    # Build conversation
    user_content = (
        f"<image> Question: Locate every person who is {action} {object_category} "
        f"and the {object_category} they interact with.\n"
        f"For each person-object pair, output bbox coordinates in JSON format like: "
        f"{{\"bbox_2d\": [x1, y1, x2, y2], \"label\": \"description\"}}. "
        f"Coordinates should be in 1000x1000 normalized format."
    )
    
    # Build response
    response_pairs = [
        {"bbox_2d": person_box, "label": "person"},
        {"bbox_2d": object_box, "label": object_category}
    ]
    response = json.dumps(response_pairs)
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": response}
    ]
    
    image_path = get_image_path(file_name, dataset_type, image_base_dir)
    
    return {
        "messages": messages,
        "images": [image_path],
        "refer_boxes": [person_box, object_box, interaction_box],
        "task_type": "grounding",
        "metadata": {
            "file_name": file_name,
            "action": action,
            "object_category": object_category,
            "dataset": dataset_type
        }
    }


# =============================================================================
# DATASET PROCESSING
# =============================================================================

def load_dataset(file_path: str) -> List[Dict]:
    """Load a simplified dataset JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)


def process_referring_dataset(
    data: List[Dict],
    dataset_type: str,
    image_base_dir: str,
) -> List[Dict]:
    """Process referring dataset and create SFT samples."""
    samples = []
    
    for item in data:
        sample = create_referring_sample(item, dataset_type, image_base_dir)
        if sample is not None:
            samples.append(sample)
    
    return samples


def process_grounding_dataset(
    data: List[Dict],
    dataset_type: str,
    image_base_dir: str,
) -> List[Dict]:
    """Process grounding dataset and create SFT samples."""
    samples = []
    
    for item in data:
        sample = create_grounding_sample(item, dataset_type, image_base_dir)
        if sample is not None:
            samples.append(sample)
    
    return samples


def balance_by_action(samples: List[Dict], max_per_action: int = 100) -> List[Dict]:
    """Balance samples by action to avoid over-representation."""
    action_groups = defaultdict(list)
    
    for sample in samples:
        action = sample["metadata"].get("action", "unknown")
        action_groups[action].append(sample)
    
    balanced = []
    for action, action_samples in action_groups.items():
        if len(action_samples) > max_per_action:
            balanced.extend(random.sample(action_samples, max_per_action))
        else:
            balanced.extend(action_samples)
    
    return balanced


def create_spatial_sft_dataset(
    input_dir: str,
    output_file: str,
    image_base_dir: str,
    num_samples: int = 5000,
    referring_ratio: float = 0.6,  # 60% referring, 40% grounding
    seed: int = 42,
) -> Dict:
    """
    Create the full spatial linking SFT dataset.
    
    Args:
        input_dir: Directory containing simplified benchmark files
        output_file: Output JSON file path
        image_base_dir: Base directory for images
        num_samples: Target number of samples
        referring_ratio: Ratio of referring vs grounding tasks
        seed: Random seed for reproducibility
        
    Returns:
        Statistics dictionary
    """
    random.seed(seed)
    input_path = Path(input_dir)
    
    all_samples = []
    stats = {
        "hico_referring": 0,
        "hico_grounding": 0,
        "swig_referring": 0,
        "swig_grounding": 0,
    }
    
    # Process HICO datasets
    hico_referring_file = input_path / "hico_referring_train_simplified.json"
    hico_grounding_file = input_path / "hico_ground_train_simplified.json"
    
    if hico_referring_file.exists():
        logger.info(f"Processing HICO referring: {hico_referring_file}")
        data = load_dataset(str(hico_referring_file))
        samples = process_referring_dataset(data, "hico", image_base_dir)
        stats["hico_referring"] = len(samples)
        all_samples.extend(samples)
        logger.info(f"  Generated {len(samples)} referring samples")
    
    if hico_grounding_file.exists():
        logger.info(f"Processing HICO grounding: {hico_grounding_file}")
        data = load_dataset(str(hico_grounding_file))
        samples = process_grounding_dataset(data, "hico", image_base_dir)
        stats["hico_grounding"] = len(samples)
        all_samples.extend(samples)
        logger.info(f"  Generated {len(samples)} grounding samples")
    
    # Process SWIG datasets
    swig_referring_file = input_path / "swig_referring_train_simplified.json"
    swig_grounding_file = input_path / "swig_ground_train_simplified.json"
    
    if swig_referring_file.exists():
        logger.info(f"Processing SWIG referring: {swig_referring_file}")
        data = load_dataset(str(swig_referring_file))
        samples = process_referring_dataset(data, "swig", image_base_dir)
        stats["swig_referring"] = len(samples)
        all_samples.extend(samples)
        logger.info(f"  Generated {len(samples)} referring samples")
    
    if swig_grounding_file.exists():
        logger.info(f"Processing SWIG grounding: {swig_grounding_file}")
        data = load_dataset(str(swig_grounding_file))
        samples = process_grounding_dataset(data, "swig", image_base_dir)
        stats["swig_grounding"] = len(samples)
        all_samples.extend(samples)
        logger.info(f"  Generated {len(samples)} grounding samples")
    
    logger.info(f"Total samples before filtering: {len(all_samples)}")
    
    # Balance by action
    all_samples = balance_by_action(all_samples, max_per_action=200)
    logger.info(f"After balancing: {len(all_samples)}")
    
    # Split by task type
    referring_samples = [s for s in all_samples if s["task_type"] == "referring"]
    grounding_samples = [s for s in all_samples if s["task_type"] == "grounding"]
    
    # Calculate target counts
    num_referring = int(num_samples * referring_ratio)
    num_grounding = num_samples - num_referring
    
    # Sample
    if len(referring_samples) > num_referring:
        referring_samples = random.sample(referring_samples, num_referring)
    if len(grounding_samples) > num_grounding:
        grounding_samples = random.sample(grounding_samples, num_grounding)
    
    final_samples = referring_samples + grounding_samples
    random.shuffle(final_samples)
    
    # Save
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(final_samples, f, indent=2)
    
    # Update stats
    stats["total"] = len(final_samples)
    stats["referring_final"] = len(referring_samples)
    stats["grounding_final"] = len(grounding_samples)
    
    logger.info(f"Final dataset: {len(final_samples)} samples")
    logger.info(f"  Referring: {len(referring_samples)}")
    logger.info(f"  Grounding: {len(grounding_samples)}")
    logger.info(f"Saved to: {output_file}")
    
    # Save stats
    stats_file = output_path.with_suffix('.stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create spatial linking SFT dataset from HOI benchmarks"
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
        default="/dataset/spatial_sft_data.json",
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
        "--referring-ratio",
        type=float,
        default=0.6,
        help="Ratio of referring tasks (default: 0.6)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    
    args = parser.parse_args()
    
    stats = create_spatial_sft_dataset(
        input_dir=args.input_dir,
        output_file=args.output_file,
        image_base_dir=args.image_base_dir,
        num_samples=args.num_samples,
        referring_ratio=args.referring_ratio,
        seed=args.seed,
    )
    
    print("\n" + "=" * 60)
    print("SPATIAL SFT DATASET CREATION COMPLETE")
    print("=" * 60)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
