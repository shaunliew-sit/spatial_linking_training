#!/usr/bin/env python3
"""
Add refer_boxes field to hoi_cof_sft dataset for spatial linking training.

This script:
1. Parses person/object boxes from user message text
2. Computes interaction box (union of person + object)
3. Adds refer_boxes field to each sample
4. Saves as new JSON file

Usage:
    python scripts/add_refer_boxes.py \
        --input /dataset/hoi_cof_sft/hoi_cof_sft_data.json \
        --output /dataset/hoi_cof_sft/hoi_cof_sft_data_with_boxes.json
"""

import json
import re
import argparse
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_interaction_box(person_box: List[int], object_box: List[int]) -> List[int]:
    """Compute union box (interaction region) from person and object boxes."""
    return [
        min(person_box[0], object_box[0]),  # x1
        min(person_box[1], object_box[1]),  # y1
        max(person_box[2], object_box[2]),  # x2
        max(person_box[3], object_box[3])   # y2
    ]


def parse_boxes_from_referring_message(content: str) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """
    Parse person and object boxes from referring task message.
    
    Format: "The person is located at [x1, y1, x2, y2] and the object is at [x1, y1, x2, y2]."
    """
    # Pattern for referring task: person at [...] and object at [...]
    pattern = r'person is located at \[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\].*?object is at \[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    match = re.search(pattern, content, re.IGNORECASE)
    
    if match:
        person_box = [int(match.group(i)) for i in range(1, 5)]
        object_box = [int(match.group(i)) for i in range(5, 9)]
        return person_box, object_box
    
    return None, None


def parse_boxes_from_grounding_answer(content: str) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    """
    Parse person and object boxes from grounding task answer.
    
    Format: [{"bbox_2d": [...], "label": "person"}, {"bbox_2d": [...], "label": "..."}]
    """
    try:
        # Find JSON array in answer
        answer_match = re.search(r'<answer>\s*(\[.*?\])\s*</answer>', content, re.DOTALL)
        if not answer_match:
            return None, None
        
        boxes_json = answer_match.group(1)
        boxes_list = json.loads(boxes_json)
        
        person_box = None
        object_box = None
        
        for item in boxes_list:
            if isinstance(item, dict) and 'bbox_2d' in item:
                bbox = item['bbox_2d']
                label = item.get('label', '').lower()
                
                if label == 'person' and person_box is None:
                    person_box = [int(x) for x in bbox]
                elif label != 'person' and object_box is None:
                    object_box = [int(x) for x in bbox]
                
                if person_box and object_box:
                    break
        
        return person_box, object_box
    except (json.JSONDecodeError, KeyError, TypeError):
        return None, None


def extract_boxes_from_sample(sample: Dict) -> Optional[List[List[int]]]:
    """
    Extract person, object, and interaction boxes from a sample.
    
    Returns:
        List of [person_box, object_box, interaction_box] or None if extraction fails
    """
    messages = sample.get('messages', [])
    
    person_box = None
    object_box = None
    
    for msg in messages:
        content = msg.get('content', '')
        role = msg.get('role', '')
        
        # Try parsing from user message (referring task)
        if role == 'user' and 'person is located at' in content:
            person_box, object_box = parse_boxes_from_referring_message(content)
            if person_box and object_box:
                break
        
        # Try parsing from assistant answer (grounding task)
        if role == 'assistant' and '<answer>' in content and 'bbox_2d' in content:
            person_box, object_box = parse_boxes_from_grounding_answer(content)
            if person_box and object_box:
                break
    
    if person_box and object_box:
        interaction_box = compute_interaction_box(person_box, object_box)
        return [person_box, object_box, interaction_box]
    
    return None


def process_dataset(input_path: str, output_path: str) -> Dict:
    """
    Process dataset to add refer_boxes field.
    
    Returns:
        Statistics dict
    """
    logger.info(f"Loading dataset from {input_path}")
    
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Processing {len(data)} samples...")
    
    stats = {
        'total': len(data),
        'with_boxes': 0,
        'without_boxes': 0,
        'referring_samples': 0,
        'grounding_samples': 0,
    }
    
    processed_data = []
    
    for sample in tqdm(data, desc="Adding refer_boxes"):
        # Try to extract boxes
        refer_boxes = extract_boxes_from_sample(sample)
        
        # Create new sample with refer_boxes
        new_sample = sample.copy()
        
        if refer_boxes:
            new_sample['refer_boxes'] = refer_boxes
            stats['with_boxes'] += 1
            
            # Determine task type
            user_content = ''
            for msg in sample.get('messages', []):
                if msg.get('role') == 'user':
                    user_content = msg.get('content', '')
                    break
            
            if 'person is located at' in user_content:
                stats['referring_samples'] += 1
            elif 'Locate every person' in user_content:
                stats['grounding_samples'] += 1
        else:
            stats['without_boxes'] += 1
        
        processed_data.append(new_sample)
    
    # Save processed data
    logger.info(f"Saving processed dataset to {output_path}")
    
    with open(output_path, 'w') as f:
        json.dump(processed_data, f, indent=2)
    
    # Save statistics
    stats_path = Path(output_path).parent / 'statistics_with_boxes.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Add refer_boxes to HOI CoF SFT dataset")
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        default='/workspace/dataset/hoi_cof_sft/hoi_cof_sft_data.json',
        help='Input JSON file path'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='/workspace/dataset/hoi_cof_sft/hoi_cof_sft_data_with_boxes.json',
        help='Output JSON file path'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Process dataset
    stats = process_dataset(args.input, args.output)
    
    # Print summary
    logger.info("=" * 50)
    logger.info("Processing Complete!")
    logger.info("=" * 50)
    logger.info(f"Total samples: {stats['total']}")
    logger.info(f"Samples with refer_boxes: {stats['with_boxes']} ({100*stats['with_boxes']/stats['total']:.1f}%)")
    logger.info(f"Samples without refer_boxes: {stats['without_boxes']}")
    logger.info(f"Referring task samples: {stats['referring_samples']}")
    logger.info(f"Grounding task samples: {stats['grounding_samples']}")
    logger.info(f"Output saved to: {args.output}")


if __name__ == '__main__':
    main()
