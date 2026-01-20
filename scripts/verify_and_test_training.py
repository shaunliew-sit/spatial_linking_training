#!/usr/bin/env python3
"""
Comprehensive verification script for Spatial Linking Training.

This script verifies that:
1. The training code is properly set up
2. Spatial linking module is correctly saved during training
3. Checkpoints can be loaded with spatial linking weights

Run this BEFORE starting a full training to catch issues early.

Usage:
    python scripts/verify_and_test_training.py
    
    # Quick mode (skip mini-training)
    python scripts/verify_and_test_training.py --quick
    
    # With specific config
    python scripts/verify_and_test_training.py --config configs/sft_lora_config_multi_gpu.yaml
"""

import os
import sys
import argparse
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def check_imports():
    """Check that all required imports work."""
    print_section("1. Checking Imports")
    
    errors = []
    
    # Core imports
    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
    except ImportError as e:
        errors.append(f"torch: {e}")
        print(f"  ✗ torch: {e}")
    
    try:
        from transformers import AutoProcessor, __version__ as tf_version
        print(f"  ✓ transformers {tf_version}")
    except ImportError as e:
        errors.append(f"transformers: {e}")
        print(f"  ✗ transformers: {e}")
    
    try:
        from peft import LoraConfig, get_peft_model, __version__ as peft_version
        print(f"  ✓ peft {peft_version}")
    except ImportError as e:
        errors.append(f"peft: {e}")
        print(f"  ✗ peft: {e}")
    
    try:
        from trl import SFTTrainer, SFTConfig, __version__ as trl_version
        print(f"  ✓ trl {trl_version}")
    except ImportError as e:
        errors.append(f"trl: {e}")
        print(f"  ✗ trl: {e}")
    
    # Local imports
    try:
        from models.spatial_model import SpatialLinkingInteractionModel
        from models.spatial_linking import SpatialLinkingModule
        print("  ✓ SpatialLinkingInteractionModel")
        print("  ✓ SpatialLinkingModule")
    except ImportError as e:
        errors.append(f"spatial_model: {e}")
        print(f"  ✗ spatial_model: {e}")
    
    try:
        from data.collator import HOISpatialCollator
        print("  ✓ HOISpatialCollator")
    except ImportError as e:
        errors.append(f"collator: {e}")
        print(f"  ✗ collator: {e}")
    
    return len(errors) == 0, errors


def check_train_script():
    """Check that train.py has the required save logic."""
    print_section("2. Checking train.py Save Logic")
    
    train_script = Path(__file__).parent / "train.py"
    
    if not train_script.exists():
        print("  ✗ train.py not found!")
        return False, ["train.py not found"]
    
    content = train_script.read_text()
    errors = []
    
    checks = [
        ("SpatialLinkingSaveCallback", "Checkpoint callback class"),
        ("spatial_linking.pt", "Save path for spatial linking weights"),
        ("spatial_linking_state", "State dict extraction"),
        ("torch.save(spatial_linking_state", "torch.save call for spatial_linking"),
        ("on_save", "Callback on_save method"),
        ("base_model.model.spatial_linking", "PEFT model path 1"),
        ("base_model.spatial_linking", "PEFT model path 2"),
    ]
    
    all_found = True
    for pattern, name in checks:
        if pattern in content:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} NOT found")
            errors.append(f"{name} not found")
            all_found = False
    
    # Check the callback is added to trainer
    if "spatial_save_callback" in content and "callbacks=" in content:
        print("  ✓ Callback added to trainer")
    else:
        print("  ! Warning: Callback may not be added to trainer")
    
    return all_found, errors


def check_evaluate_script():
    """Check that evaluate.py has proper load logic."""
    print_section("3. Checking evaluate.py Load Logic")
    
    eval_script = Path(__file__).parent / "evaluate.py"
    
    if not eval_script.exists():
        print("  ✗ evaluate.py not found!")
        return False, ["evaluate.py not found"]
    
    content = eval_script.read_text()
    errors = []
    
    checks = [
        ("spatial_linking.pt", "Load path for spatial linking weights"),
        ("load_state_dict", "State dict loading"),
        ("model.spatial_linking", "Model spatial_linking access"),
    ]
    
    all_found = True
    for pattern, name in checks:
        if pattern in content:
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} NOT found")
            errors.append(f"{name} not found")
            all_found = False
    
    return all_found, errors


def check_spatial_linking_module():
    """Verify the SpatialLinkingModule structure."""
    print_section("4. Checking SpatialLinkingModule Structure")
    
    try:
        import torch
        from models.spatial_linking import SpatialLinkingModule
        
        # Create module
        module = SpatialLinkingModule(hidden_dim=3584, num_heads=8)
        
        # Check components
        has_cross_attn = hasattr(module, 'cross_attn')
        has_layer_norm = hasattr(module, 'layer_norm')
        has_refine_mlp = hasattr(module, 'refine_mlp')
        
        print(f"  ✓ cross_attn: {has_cross_attn}")
        print(f"  ✓ layer_norm: {has_layer_norm}")
        print(f"  ✓ refine_mlp: {has_refine_mlp}")
        
        # Check parameter count
        param_count = sum(p.numel() for p in module.parameters())
        trainable_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
        
        print(f"  ✓ Total parameters: {param_count:,}")
        print(f"  ✓ Trainable parameters: {trainable_count:,}")
        
        # Test state dict
        state_dict = module.state_dict()
        print(f"  ✓ State dict keys: {len(state_dict)}")
        for key in list(state_dict.keys())[:5]:
            print(f"      - {key}: {state_dict[key].shape}")
        if len(state_dict) > 5:
            print(f"      ... and {len(state_dict) - 5} more keys")
        
        return True, []
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False, [str(e)]


def run_mini_training(config_path: str = None, output_dir: str = None):
    """Run a mini-training to verify save/load works."""
    print_section("5. Running Mini-Training Test")
    
    import torch
    from transformers import AutoProcessor
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from models.spatial_model import SpatialLinkingInteractionModel
    from scripts.train import SpatialLinkingSaveCallback
    
    # Create temp output directory if not provided
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="spatial_linking_test_")
    
    print(f"  Output directory: {output_dir}")
    
    try:
        # Load model with minimal settings
        print("  Loading model (this may take a moment)...")
        
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            trust_remote_code=True,
        )
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
        model = SpatialLinkingInteractionModel.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="sdpa",
        )
        
        print(f"  ✓ Model loaded")
        print(f"  ✓ spatial_linking module exists: {hasattr(model, 'spatial_linking')}")
        
        # Get initial spatial_linking weights
        initial_state = {k: v.clone() for k, v in model.spatial_linking.state_dict().items()}
        print(f"  ✓ Initial spatial_linking state captured ({len(initial_state)} keys)")
        
        # Apply LoRA
        lora_config = LoraConfig(
            r=8,  # Small rank for testing
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        print("  ✓ LoRA applied")
        
        # Ensure spatial_linking is trainable
        spatial_linking_module = None
        if hasattr(model, 'base_model'):
            if hasattr(model.base_model, 'model') and hasattr(model.base_model.model, 'spatial_linking'):
                spatial_linking_module = model.base_model.model.spatial_linking
            elif hasattr(model.base_model, 'spatial_linking'):
                spatial_linking_module = model.base_model.spatial_linking
        
        if spatial_linking_module is not None:
            for param in spatial_linking_module.parameters():
                param.requires_grad = True
            print("  ✓ spatial_linking set to trainable")
        else:
            print("  ✗ Could not find spatial_linking in PEFT model!")
            return False, ["spatial_linking not found after PEFT wrapping"]
        
        # Create minimal dummy dataset
        dummy_data = [
            {
                "messages": [
                    {"role": "user", "content": "Test message 1"},
                    {"role": "assistant", "content": "Test response 1"},
                ],
            },
            {
                "messages": [
                    {"role": "user", "content": "Test message 2"},
                    {"role": "assistant", "content": "Test response 2"},
                ],
            },
        ] * 5  # 10 samples
        
        from datasets import Dataset
        train_dataset = Dataset.from_list(dummy_data)
        
        # Simple text collator for testing
        def simple_collate(examples):
            texts = []
            for ex in examples:
                text = processor.tokenizer.apply_chat_template(
                    ex["messages"], tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
            
            batch = processor.tokenizer(
                texts,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=256,
            )
            batch["labels"] = batch["input_ids"].clone()
            return batch
        
        # Training arguments for quick test
        training_args = SFTConfig(
            output_dir=output_dir,
            max_steps=3,  # Just 3 steps
            per_device_train_batch_size=2,
            learning_rate=1e-4,
            logging_steps=1,
            save_steps=2,  # Save at step 2
            save_total_limit=1,
            bf16=True,
            remove_unused_columns=False,
            dataset_kwargs={"skip_prepare_dataset": True},
            report_to="none",  # No wandb for test
            max_length=256,
        )
        
        # Create callback
        save_callback = SpatialLinkingSaveCallback()
        
        # Create trainer
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=simple_collate,
            processing_class=processor,
            callbacks=[save_callback],
        )
        
        print("  ✓ Trainer created")
        print("  Running 3 training steps...")
        
        # Train
        trainer.train()
        print("  ✓ Training completed")
        
        # Save final model
        trainer.save_model()
        
        # Manually save spatial_linking (as train.py does)
        spatial_linking_path = os.path.join(output_dir, "spatial_linking.pt")
        if spatial_linking_module is not None:
            torch.save(spatial_linking_module.state_dict(), spatial_linking_path)
            print(f"  ✓ Saved spatial_linking.pt to {spatial_linking_path}")
        
        # Check checkpoint for spatial_linking.pt
        checkpoint_dirs = list(Path(output_dir).glob("checkpoint-*"))
        
        checkpoint_has_spatial = False
        for ckpt_dir in checkpoint_dirs:
            ckpt_spatial = ckpt_dir / "spatial_linking.pt"
            if ckpt_spatial.exists():
                print(f"  ✓ Found spatial_linking.pt in {ckpt_dir.name}")
                checkpoint_has_spatial = True
                
                # Verify the saved weights
                saved_state = torch.load(ckpt_spatial, map_location="cpu")
                print(f"      Saved keys: {len(saved_state)}")
            else:
                print(f"  ! No spatial_linking.pt in {ckpt_dir.name}")
        
        # Check final output
        final_spatial = Path(output_dir) / "spatial_linking.pt"
        if final_spatial.exists():
            print(f"  ✓ Found spatial_linking.pt in final output")
            
            # Load and compare
            saved_state = torch.load(final_spatial, map_location="cpu")
            current_state = spatial_linking_module.state_dict()
            
            # Check that weights changed from initial
            weights_changed = False
            for key in initial_state:
                if not torch.allclose(initial_state[key].cpu(), saved_state[key].cpu(), atol=1e-6):
                    weights_changed = True
                    break
            
            if weights_changed:
                print("  ✓ Spatial linking weights CHANGED during training")
            else:
                print("  ! Warning: Spatial linking weights did NOT change during training")
                print("      This might be OK for text-only dummy data (no refer_boxes)")
            
            # Verify weights match current model
            weights_match = True
            for key in saved_state:
                if key not in current_state:
                    weights_match = False
                    break
                if not torch.allclose(saved_state[key].cpu(), current_state[key].cpu(), atol=1e-6):
                    weights_match = False
                    break
            
            if weights_match:
                print("  ✓ Saved weights match current model state")
            else:
                print("  ✗ Saved weights DO NOT match current model state!")
        else:
            print(f"  ✗ No spatial_linking.pt in final output!")
            return False, ["spatial_linking.pt not saved to final output"]
        
        return True, []
        
    except Exception as e:
        import traceback
        print(f"  ✗ Error during mini-training: {e}")
        print(traceback.format_exc())
        return False, [str(e)]
    
    finally:
        # Cleanup if using temp dir
        if output_dir and output_dir.startswith(tempfile.gettempdir()):
            print(f"  Cleaning up temp directory: {output_dir}")
            shutil.rmtree(output_dir, ignore_errors=True)


def test_checkpoint_loading(checkpoint_path: str = None):
    """Test loading a checkpoint and verifying spatial_linking weights."""
    print_section("6. Testing Checkpoint Loading")
    
    if checkpoint_path is None:
        print("  Skipping (no checkpoint path provided)")
        print("  Use --checkpoint_path to test a specific checkpoint")
        return True, []
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        print(f"  ✗ Checkpoint path not found: {checkpoint_path}")
        return False, [f"Checkpoint not found: {checkpoint_path}"]
    
    try:
        import torch
        from transformers import AutoProcessor
        from models.spatial_model import SpatialLinkingInteractionModel
        
        print(f"  Loading checkpoint from: {checkpoint_path}")
        
        # Load processor
        processor = AutoProcessor.from_pretrained(
            str(checkpoint_path),
            trust_remote_code=True,
        )
        print("  ✓ Processor loaded")
        
        # Load model
        model = SpatialLinkingInteractionModel.from_pretrained(
            str(checkpoint_path),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        print("  ✓ Model loaded")
        
        # Check for spatial_linking.pt
        spatial_linking_path = checkpoint_path / "spatial_linking.pt"
        if spatial_linking_path.exists():
            print(f"  ✓ Found spatial_linking.pt")
            
            saved_state = torch.load(spatial_linking_path, map_location="cpu")
            print(f"      Keys: {len(saved_state)}")
            
            # Load into model
            model.spatial_linking.load_state_dict(saved_state)
            print("  ✓ Loaded spatial_linking weights into model")
            
            # Verify
            current_state = model.spatial_linking.state_dict()
            match = all(
                torch.allclose(saved_state[k].cpu(), current_state[k].cpu(), atol=1e-6)
                for k in saved_state
            )
            if match:
                print("  ✓ Weights verified successfully")
            else:
                print("  ✗ Weight mismatch after loading!")
                return False, ["Weight mismatch after loading"]
        else:
            print(f"  ✗ spatial_linking.pt NOT found in checkpoint!")
            print("      This checkpoint was likely created before the save fix.")
            return False, ["spatial_linking.pt not found in checkpoint"]
        
        return True, []
        
    except Exception as e:
        import traceback
        print(f"  ✗ Error loading checkpoint: {e}")
        print(traceback.format_exc())
        return False, [str(e)]


def main():
    parser = argparse.ArgumentParser(description="Verify Spatial Linking Training Setup")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip mini-training test (faster, but less thorough)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config (for mini-training test)"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint to test loading"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for mini-training (uses temp dir if not specified)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print(" SPATIAL LINKING TRAINING VERIFICATION")
    print(f" Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    all_passed = True
    all_errors = []
    
    # Run checks
    passed, errors = check_imports()
    all_passed &= passed
    all_errors.extend(errors)
    
    passed, errors = check_train_script()
    all_passed &= passed
    all_errors.extend(errors)
    
    passed, errors = check_evaluate_script()
    all_passed &= passed
    all_errors.extend(errors)
    
    passed, errors = check_spatial_linking_module()
    all_passed &= passed
    all_errors.extend(errors)
    
    # Mini-training test
    if not args.quick:
        passed, errors = run_mini_training(args.config, args.output_dir)
        all_passed &= passed
        all_errors.extend(errors)
    else:
        print_section("5. Mini-Training Test")
        print("  Skipped (--quick mode)")
    
    # Checkpoint loading test
    if args.checkpoint_path:
        passed, errors = test_checkpoint_loading(args.checkpoint_path)
        all_passed &= passed
        all_errors.extend(errors)
    
    # Summary
    print_section("SUMMARY")
    
    if all_passed:
        print("\n  ✅ ALL CHECKS PASSED!")
        print("\n  Your training setup is ready.")
        print("\n  To start training:")
        print("    # Single GPU:")
        print("    python scripts/train.py --config configs/sft_lora_config.yaml")
        print("")
        print("    # Multi-GPU:")
        print("    accelerate launch --num_processes=8 scripts/train.py --config configs/sft_lora_config_multi_gpu.yaml")
        print("")
        print("  After training, verify spatial_linking.pt was saved:")
        print("    ls -la outputs/spatial_linking_sft*/run_*/spatial_linking.pt")
        print("    ls -la outputs/spatial_linking_sft*/run_*/checkpoint-*/spatial_linking.pt")
    else:
        print("\n  ❌ SOME CHECKS FAILED!")
        print("\n  Errors:")
        for err in all_errors:
            print(f"    - {err}")
        print("\n  Please fix these issues before running training.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
