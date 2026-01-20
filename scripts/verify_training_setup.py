#!/usr/bin/env python3
"""
Quick verification script to ensure training setup is correct.
Run this before starting training to catch issues early.
"""

import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    errors = []
    warnings = []
    
    print("=" * 60)
    print("Spatial Linking Training Setup Verification")
    print("=" * 60)
    
    # 1. Check imports
    print("\n[1/5] Checking imports...")
    try:
        import torch
        from transformers import AutoProcessor
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer, SFTConfig
        from models.spatial_model import SpatialLinkingInteractionModel
        from data.collator import HOISpatialCollator
        print("  ✓ All imports successful")
    except ImportError as e:
        errors.append(f"Import error: {e}")
        print(f"  ✗ Import failed: {e}")
    
    # 2. Check SpatialLinkingInteractionModel has spatial_linking
    print("\n[2/5] Checking SpatialLinkingInteractionModel structure...")
    try:
        from models.spatial_model import SpatialLinkingInteractionModel
        import inspect
        source = inspect.getsource(SpatialLinkingInteractionModel.__init__)
        if "spatial_linking" in source:
            print("  ✓ spatial_linking module is defined in __init__")
        else:
            warnings.append("spatial_linking not explicitly set in __init__")
            print("  ! Warning: spatial_linking not found in __init__ (may be defined elsewhere)")
    except Exception as e:
        warnings.append(f"Could not inspect model: {e}")
        print(f"  ! Warning: {e}")
    
    # 3. Check training script has spatial_linking save logic
    print("\n[3/5] Checking train.py save logic...")
    train_script = Path(__file__).parent / "train.py"
    if train_script.exists():
        content = train_script.read_text()
        
        checks = [
            ("SpatialLinkingSaveCallback", "Checkpoint callback"),
            ("spatial_linking.pt", "Spatial linking save path"),
            ("spatial_linking_state", "State dict extraction"),
            ("torch.save", "torch.save call"),
        ]
        
        all_found = True
        for pattern, name in checks:
            if pattern in content:
                print(f"  ✓ {name} found")
            else:
                errors.append(f"{name} not found in train.py")
                print(f"  ✗ {name} NOT found")
                all_found = False
        
        if all_found:
            print("  ✓ All save logic present")
    else:
        errors.append("train.py not found")
        print("  ✗ train.py not found")
    
    # 4. Check evaluate.py has spatial_linking load logic
    print("\n[4/5] Checking evaluate.py load logic...")
    eval_script = Path(__file__).parent / "evaluate.py"
    if eval_script.exists():
        content = eval_script.read_text()
        
        if "spatial_linking.pt" in content and "load_state_dict" in content:
            print("  ✓ Spatial linking load logic present")
        else:
            errors.append("evaluate.py missing spatial_linking load logic")
            print("  ✗ Spatial linking load logic NOT found")
    else:
        errors.append("evaluate.py not found")
        print("  ✗ evaluate.py not found")
    
    # 5. Check PEFT model structure handling
    print("\n[5/5] Checking PEFT model structure handling...")
    train_content = train_script.read_text() if train_script.exists() else ""
    
    peft_checks = [
        "base_model.model.spatial_linking",
        "base_model.spatial_linking",
        "model.spatial_linking",
    ]
    
    found = [p for p in peft_checks if p in train_content]
    if len(found) >= 2:
        print("  ✓ Multiple PEFT model paths handled")
    else:
        warnings.append("Limited PEFT model path handling")
        print("  ! Only found: " + ", ".join(found) if found else "None")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if errors:
        print(f"\n❌ ERRORS ({len(errors)}):")
        for e in errors:
            print(f"   - {e}")
    
    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"   - {w}")
    
    if not errors and not warnings:
        print("\n✅ All checks passed! Ready for training.")
    elif not errors:
        print("\n✅ No critical errors. Ready for training (with warnings).")
    else:
        print("\n❌ Please fix errors before training.")
        return 1
    
    # Training command
    print("\n" + "=" * 60)
    print("TRAINING COMMAND")
    print("=" * 60)
    print("""
# Run training:
python scripts/train.py --config configs/sft_lora_config.yaml

# After training, verify spatial_linking.pt was saved:
ls -la outputs/spatial_linking_sft/run_*/spatial_linking.pt

# Then evaluate:
python scripts/evaluate.py \\
    --model_path outputs/spatial_linking_sft/run_XXXXX \\
    --test_file /path/to/test.json \\
    --task_type referring \\
    --max_samples 20 \\
    --verbose
""")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
