"""
Quick memory-efficient test for fusion model training fix.
Tests forward pass, loss computation, and ensures output format is correct.
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.yolo import FusionModel
from utils.loss import ComputeLoss
from utils.fusion_dataloaders import create_fusion_dataloader

def test_fusion_model_output():
    """Test that fusion model returns correct output format for training and eval."""
    print("="*60)
    print("Testing Fusion Model Output Format")
    print("="*60)
    
    # Create small model
    print("\n1. Creating fusion model...")
    model = FusionModel(cfg='models/yolov5n_fusion.yaml', ch=3, nc=1)
    
    # Create dummy inputs
    batch_size = 2
    img_size = 416  # Smaller size to save memory
    rgb_input = torch.randn(batch_size, 3, img_size, img_size)
    ir_input = torch.randn(batch_size, 3, img_size, img_size)
    
    print(f"   RGB input shape: {rgb_input.shape}")
    print(f"   IR input shape: {ir_input.shape}")
    
    # Test training mode
    print("\n2. Testing TRAINING mode output...")
    model.train()
    with torch.no_grad():
        output_train = model(rgb_input, ir_input)
    
    print(f"   Output type: {type(output_train)}")
    if isinstance(output_train, list):
        print(f"   Output length: {len(output_train)}")
        for i, pred in enumerate(output_train):
            print(f"   Prediction {i} shape: {pred.shape}")
    else:
        print(f"   ❌ FAILED: Training mode should return list, got {type(output_train)}")
        return False
    
    if len(output_train) != 3:
        print(f"   ❌ FAILED: Expected 3 predictions, got {len(output_train)}")
        return False
    
    # Test eval mode
    print("\n3. Testing EVAL mode output...")
    model.eval()
    with torch.no_grad():
        output_eval = model(rgb_input, ir_input)
    
    print(f"   Output type: {type(output_eval)}")
    if isinstance(output_eval, tuple) and len(output_eval) == 2:
        decoded_preds, raw_preds = output_eval
        print(f"   ✓ Eval mode returns tuple (decoded, raw)")
        print(f"   Decoded predictions shape: {decoded_preds.shape}")
        print(f"   Raw predictions: list of {len(raw_preds)} tensors")
        for i, p in enumerate(raw_preds):
            print(f"     Raw {i} shape: {p.shape}")
    elif isinstance(output_eval, list):
        print(f"   ⚠️  WARNING: Eval mode returned list instead of tuple")
        print(f"   This might work but is not ideal")
    else:
        print(f"   ❌ FAILED: Eval mode should return tuple, got {type(output_eval)}")
        return False
    
    print("\n   ✅ Output formats are correct!")
    return True


def test_loss_computation():
    """Test that ComputeLoss can process fusion model output."""
    print("\n" + "="*60)
    print("Testing Loss Computation")
    print("="*60)
    
    # Create model and loss
    print("\n1. Creating model and loss function...")
    model = FusionModel(cfg='models/yolov5n_fusion.yaml', ch=3, nc=1)
    model.train()
    
    # Set model hyperparameters (required for ComputeLoss)
    model.hyp = {
        'box': 0.05,
        'cls': 0.5,
        'cls_pw': 1.0,
        'obj': 1.0,
        'obj_pw': 1.0,
        'label_smoothing': 0.0,
        'fl_gamma': 0.0,
    }
    model.nc = 1
    
    compute_loss = ComputeLoss(model)
    
    # Create dummy inputs and targets
    batch_size = 2
    img_size = 416
    rgb_input = torch.randn(batch_size, 3, img_size, img_size)
    ir_input = torch.randn(batch_size, 3, img_size, img_size)
    
    # Create dummy targets: (image_idx, class, x, y, w, h) normalized
    targets = torch.tensor([
        [0, 0, 0.5, 0.5, 0.2, 0.2],  # Object in image 0
        [1, 0, 0.3, 0.3, 0.15, 0.15],  # Object in image 1
    ], dtype=torch.float32)
    
    print(f"   Targets shape: {targets.shape}")
    
    # Forward pass
    print("\n2. Running forward pass...")
    with torch.no_grad():
        pred = model(rgb_input, ir_input)
    
    print(f"   Predictions type: {type(pred)}")
    if isinstance(pred, list):
        print(f"   Number of prediction scales: {len(pred)}")
        for i, p in enumerate(pred):
            print(f"   Scale {i} shape: {p.shape}")
    
    # Compute loss
    print("\n3. Computing loss...")
    try:
        loss, loss_items = compute_loss(pred, targets)
        print(f"   ✓ Loss computation successful!")
        print(f"   Total loss: {loss.item():.4f}")
        print(f"   Loss items (box, obj, cls): {loss_items.tolist()}")
        
        # Check that loss is not zero (indicates model is working)
        if loss.item() == 0:
            print("   ⚠️  WARNING: Loss is zero - model may not be learning!")
        else:
            print("   ✓ Loss is non-zero - model output is valid!")
        
        return True
    except Exception as e:
        print(f"   ❌ FAILED: Loss computation error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_training_step():
    """Test a single training step with minimal memory usage."""
    print("\n" + "="*60)
    print("Testing Mini Training Step")
    print("="*60)
    
    # Create model
    print("\n1. Creating model...")
    model = FusionModel(cfg='models/yolov5n_fusion.yaml', ch=3, nc=1)
    model.train()
    model.hyp = {
        'box': 0.05, 'cls': 0.5, 'cls_pw': 1.0, 'obj': 1.0, 'obj_pw': 1.0,
        'label_smoothing': 0.0, 'fl_gamma': 0.0,
    }
    model.nc = 1
    model.class_weights = torch.ones(1)
    
    # Create optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.937)
    compute_loss = ComputeLoss(model)
    
    # Tiny batch
    batch_size = 1
    img_size = 320  # Even smaller to save memory
    rgb_input = torch.randn(batch_size, 3, img_size, img_size)
    ir_input = torch.randn(batch_size, 3, img_size, img_size)
    targets = torch.tensor([[0, 0, 0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)
    
    print(f"   Batch size: {batch_size}, Image size: {img_size}")
    
    # Training step
    print("\n2. Running training step...")
    optimizer.zero_grad()
    
    # Forward
    pred = model(rgb_input, ir_input)
    print(f"   Prediction type: {type(pred)}, length: {len(pred) if isinstance(pred, (list, tuple)) else 'N/A'}")
    
    # Loss
    loss, loss_items = compute_loss(pred, targets)
    print(f"   Loss: {loss.item():.4f}")
    
    # Backward
    loss.backward()
    optimizer.step()
    
    print(f"   ✓ Training step completed successfully!")
    print(f"   Final loss: {loss.item():.4f}")
    
    return True


if __name__ == "__main__":
    print("\n" + "="*60)
    print("FUSION MODEL QUICK TEST")
    print("="*60)
    print("\nThis script tests the fusion model fixes with minimal memory usage.")
    print("Image sizes are reduced to save memory.\n")
    
    results = []
    
    # Test 1: Output format
    try:
        results.append(("Output Format", test_fusion_model_output()))
    except Exception as e:
        print(f"\n❌ Test 1 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Output Format", False))
    
    # Test 2: Loss computation
    try:
        results.append(("Loss Computation", test_loss_computation()))
    except Exception as e:
        print(f"\n❌ Test 2 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Loss Computation", False))
    
    # Test 3: Training step
    try:
        results.append(("Training Step", test_mini_training_step()))
    except Exception as e:
        print(f"\n❌ Test 3 failed with exception: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Training Step", False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 All tests passed! Fusion model should work correctly now.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    sys.exit(0 if all_passed else 1)

