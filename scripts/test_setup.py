#!/usr/bin/env python3
"""
Quick test script to verify the spatial linking setup works correctly.

Tests:
1. Dataset loading
2. Collator functionality  
3. Model instantiation
4. Forward pass with refer_boxes

Usage:
    python scripts/test_setup.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
from PIL import Image

def test_dataset():
    """Test that dataset loads correctly."""
    print("=" * 50)
    print("Testing Dataset Loading...")
    print("=" * 50)
    
    dataset_path = "/workspace/dataset/hoi_cof_sft/hoi_cof_sft_data_with_boxes.json"
    
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} samples")
    
    # Check first sample
    sample = data[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Number of messages: {len(sample['messages'])}")
    print(f"Number of images: {len(sample['images'])}")
    
    if 'refer_boxes' in sample:
        print(f"refer_boxes: {sample['refer_boxes']}")
        print(f"Number of boxes: {len(sample['refer_boxes'])}")
    else:
        print("WARNING: No refer_boxes found!")
    
    # Count samples with/without boxes
    with_boxes = sum(1 for s in data if 'refer_boxes' in s and s['refer_boxes'])
    print(f"Samples with refer_boxes: {with_boxes}/{len(data)}")
    
    print("Dataset test PASSED\n")
    return data[:5]  # Return first 5 for further testing


def test_collator(samples):
    """Test the collator."""
    print("=" * 50)
    print("Testing Collator...")
    print("=" * 50)
    
    from transformers import AutoProcessor
    from data.collator import HOISpatialCollator
    
    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        trust_remote_code=True,
    )
    
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Create collator
    collator = HOISpatialCollator(
        processor=processor,
        max_length=2048,
        image_base_dir="/workspace/dataset/hoi_cof_sft/images",
    )
    
    # Test with single sample
    print("Testing single sample collation...")
    batch = collator([samples[0]])
    
    print(f"Batch keys: {batch.keys()}")
    print(f"input_ids shape: {batch['input_ids'].shape}")
    print(f"attention_mask shape: {batch['attention_mask'].shape}")
    
    if 'pixel_values' in batch:
        print(f"pixel_values shape: {batch['pixel_values'].shape}")
    
    if 'refer_boxes' in batch:
        print(f"refer_boxes: {type(batch['refer_boxes'])}")
        if isinstance(batch['refer_boxes'], list):
            print(f"  First item shape: {batch['refer_boxes'][0].shape if batch['refer_boxes'][0] is not None else None}")
    
    print("Collator test PASSED\n")
    return batch, processor


def test_model(batch, processor):
    """Test model instantiation and forward pass."""
    print("=" * 50)
    print("Testing Model...")
    print("=" * 50)
    
    from models.spatial_model import SpatialLinkingInteractionModel
    
    print("Loading model (this may take a moment)...")
    model = SpatialLinkingInteractionModel.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    
    # Set box token IDs
    model.set_box_token_ids(processor.tokenizer)
    
    print(f"Model loaded: {type(model).__name__}")
    print(f"Spatial linking trainable params: {model.spatial_linking.get_trainable_params():,}")
    
    # Check if box tokens exist
    box_end_id = processor.tokenizer.convert_tokens_to_ids("<|box_end|>")
    box_start_id = processor.tokenizer.convert_tokens_to_ids("<|box_start|>")
    print(f"box_start_token_id: {box_start_id}")
    print(f"box_end_token_id: {box_end_id}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    
    # Move batch to device
    device = next(model.parameters()).device
    
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch.get('labels')
    if labels is not None:
        labels = labels.to(device)
    
    pixel_values = batch.get('pixel_values')
    if pixel_values is not None:
        pixel_values = pixel_values.to(device, dtype=torch.bfloat16)
    
    image_grid_thw = batch.get('image_grid_thw')
    if image_grid_thw is not None:
        image_grid_thw = image_grid_thw.to(device)
    
    refer_boxes = batch.get('refer_boxes')
    
    with torch.no_grad():
        try:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                refer_boxes=refer_boxes,
                output_spatial_attentions=True,
            )
            
            print(f"Forward pass successful!")
            print(f"Loss: {outputs.loss.item() if outputs.loss is not None else 'N/A'}")
            print(f"Logits shape: {outputs.logits.shape}")
            
            # Check attention weights
            attn_weights = model.get_spatial_attention_weights()
            if attn_weights:
                print(f"Spatial attention weights captured: {len(attn_weights)} batches")
            
        except Exception as e:
            print(f"Forward pass FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("Model test PASSED\n")
    return True


def main():
    print("\n" + "=" * 60)
    print("SPATIAL LINKING TRAINING SETUP TEST")
    print("=" * 60 + "\n")
    
    try:
        # Test 1: Dataset
        samples = test_dataset()
        
        # Test 2: Collator
        batch, processor = test_collator(samples)
        
        # Test 3: Model
        success = test_model(batch, processor)
        
        if success:
            print("=" * 60)
            print("ALL TESTS PASSED!")
            print("=" * 60)
            print("\nYou can now run training with:")
            print("  python scripts/train.py --config configs/sft_lora_config_dgx.yaml")
        else:
            print("=" * 60)
            print("SOME TESTS FAILED - Please check errors above")
            print("=" * 60)
            
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
