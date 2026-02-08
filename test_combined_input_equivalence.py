"""
Test script to verify that combined input (B, 6, H, W) produces identical outputs
to dual inputs (B, 3, H, W) each, ensuring model behavior remains unchanged.
"""

import torch
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.fusion_model import FusionModel

def test_output_equivalence():
    """Test that combined input produces identical outputs to dual inputs."""
    print("="*70)
    print("Testing Combined Input Equivalence")
    print("="*70)
    
    # Create model
    print("\n1. Creating fusion model...")
    model = FusionModel(cfg='models/yolov5n_fusion.yaml', ch=3, nc=1)
    model.eval()
    
    # Create test inputs
    batch_size = 2
    img_size = 640
    print(f"\n2. Creating test inputs (batch_size={batch_size}, img_size={img_size})...")
    
    # Create RGB and IR inputs
    rgb_input = torch.randn(batch_size, 3, img_size, img_size)
    ir_input = torch.randn(batch_size, 3, img_size, img_size)
    
    # Create combined input by concatenating along channel dimension
    combined_input = torch.cat([rgb_input, ir_input], dim=1)  # (B, 6, H, W)
    
    print(f"   RGB input shape: {rgb_input.shape}")
    print(f"   IR input shape: {ir_input.shape}")
    print(f"   Combined input shape: {combined_input.shape}")
    
    # Test 1: Training mode
    print("\n3. Testing TRAINING mode...")
    model.train()
    with torch.no_grad():
        # Forward pass with dual inputs (old way - backward compatible)
        output_dual = model(rgb_input, ir_input)
        
        # Forward pass with combined input (new way)
        output_combined = model(combined_input)
    
    # Compare outputs
    if isinstance(output_dual, list) and isinstance(output_combined, list):
        if len(output_dual) == len(output_combined):
            print(f"   ✓ Both outputs are lists with {len(output_dual)} elements")
            all_match = True
            for i, (dual_pred, combined_pred) in enumerate(zip(output_dual, output_combined)):
                if torch.allclose(dual_pred, combined_pred, rtol=1e-5, atol=1e-6):
                    print(f"   ✓ Prediction {i}: Shapes match {dual_pred.shape}, values match")
                else:
                    max_diff = (dual_pred - combined_pred).abs().max().item()
                    print(f"   ✗ Prediction {i}: Values differ! Max difference: {max_diff:.2e}")
                    all_match = False
            if all_match:
                print("   ✅ TRAINING MODE: Outputs are IDENTICAL!")
            else:
                print("   ❌ TRAINING MODE: Outputs differ!")
                return False
        else:
            print(f"   ❌ Different list lengths: {len(output_dual)} vs {len(output_combined)}")
            return False
    else:
        print(f"   ❌ Unexpected output types: {type(output_dual)} vs {type(output_combined)}")
        return False
    
    # Test 2: Eval mode
    print("\n4. Testing EVAL mode...")
    model.eval()
    with torch.no_grad():
        # Forward pass with dual inputs (old way - backward compatible)
        output_dual = model(rgb_input, ir_input)
        
        # Forward pass with combined input (new way)
        output_combined = model(combined_input)
    
    # Compare outputs
    if isinstance(output_dual, tuple) and isinstance(output_combined, tuple):
        if len(output_dual) == len(output_combined) == 2:
            dual_decoded, dual_raw = output_dual
            combined_decoded, combined_raw = output_combined
            
            # Compare decoded predictions
            if torch.allclose(dual_decoded, combined_decoded, rtol=1e-5, atol=1e-6):
                print(f"   ✓ Decoded predictions match: shape {dual_decoded.shape}")
            else:
                max_diff = (dual_decoded - combined_decoded).abs().max().item()
                print(f"   ✗ Decoded predictions differ! Max difference: {max_diff:.2e}")
                return False
            
            # Compare raw predictions
            if isinstance(dual_raw, list) and isinstance(combined_raw, list):
                if len(dual_raw) == len(combined_raw):
                    all_match = True
                    for i, (dual_r, combined_r) in enumerate(zip(dual_raw, combined_raw)):
                        if torch.allclose(dual_r, combined_r, rtol=1e-5, atol=1e-6):
                            print(f"   ✓ Raw prediction {i}: Shapes match {dual_r.shape}, values match")
                        else:
                            max_diff = (dual_r - combined_r).abs().max().item()
                            print(f"   ✗ Raw prediction {i}: Values differ! Max difference: {max_diff:.2e}")
                            all_match = False
                    if all_match:
                        print("   ✅ EVAL MODE: Outputs are IDENTICAL!")
                    else:
                        print("   ❌ EVAL MODE: Outputs differ!")
                        return False
                else:
                    print(f"   ❌ Different raw list lengths: {len(dual_raw)} vs {len(combined_raw)}")
                    return False
            else:
                print(f"   ❌ Unexpected raw output types: {type(dual_raw)} vs {type(combined_raw)}")
                return False
        else:
            print(f"   ❌ Unexpected tuple lengths: {len(output_dual)} vs {len(output_combined)}")
            return False
    elif isinstance(output_dual, list) and isinstance(output_combined, list):
        # Some models return list in eval mode
        if len(output_dual) == len(output_combined):
            all_match = True
            for i, (dual_pred, combined_pred) in enumerate(zip(output_dual, output_combined)):
                if torch.allclose(dual_pred, combined_pred, rtol=1e-5, atol=1e-6):
                    print(f"   ✓ Prediction {i}: Shapes match {dual_pred.shape}, values match")
                else:
                    max_diff = (dual_pred - combined_pred).abs().max().item()
                    print(f"   ✗ Prediction {i}: Values differ! Max difference: {max_diff:.2e}")
                    all_match = False
            if all_match:
                print("   ✅ EVAL MODE: Outputs are IDENTICAL!")
            else:
                print("   ❌ EVAL MODE: Outputs differ!")
                return False
        else:
            print(f"   ❌ Different list lengths: {len(output_dual)} vs {len(output_combined)}")
            return False
    else:
        print(f"   ❌ Unexpected output types: {type(output_dual)} vs {type(output_combined)}")
        return False
    
    print("\n" + "="*70)
    print("✅ ALL TESTS PASSED: Model outputs are IDENTICAL!")
    print("="*70)
    print("\nThe combined input approach produces exactly the same outputs")
    print("as the dual input approach, confirming model behavior is unchanged.")
    return True


if __name__ == "__main__":
    try:
        success = test_output_equivalence()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
