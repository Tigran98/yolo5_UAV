"""
Quick debug script to check fusion model output format
"""

import torch
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from models.yolo import FusionModel

print("Creating fusion model...")
model = FusionModel(cfg='models/yolov5n_fusion.yaml', ch=3, nc=1)

# Create dummy inputs
batch_size = 1
img_size = 256
rgb = torch.randn(batch_size, 3, img_size, img_size)
ir = torch.randn(batch_size, 3, img_size, img_size)

print(f"\nInput shapes: RGB {rgb.shape}, IR {ir.shape}")

# Test training mode
print("\n" + "="*60)
print("TRAINING MODE")
print("="*60)
model.train()
print(f"Model training mode: {model.training}")
print(f"Detect layer training mode: {model.head[-1].training}")

with torch.no_grad():
    output = model(rgb, ir)

print(f"Output type: {type(output)}")
if isinstance(output, list):
    print(f"Output length: {len(output)}")
    for i, p in enumerate(output):
        print(f"  Prediction {i}: shape {p.shape}, dtype {p.dtype}")
elif isinstance(output, tuple):
    print(f"Output length: {len(output)}")
    for i, o in enumerate(output):
        print(f"  Element {i}: type {type(o)}")
        if isinstance(o, list):
            print(f"    List length: {len(o)}")
            for j, p in enumerate(o):
                print(f"      Item {j}: shape {p.shape}")
        elif isinstance(o, torch.Tensor):
            print(f"    Tensor shape: {o.shape}")
else:
    print(f"  Single output: shape {output.shape if hasattr(output, 'shape') else 'N/A'}")

# Test eval mode
print("\n" + "="*60)
print("EVAL MODE")
print("="*60)
model.eval()
print(f"Model training mode: {model.training}")
print(f"Detect layer training mode: {model.head[-1].training}")

with torch.no_grad():
    output = model(rgb, ir)

print(f"Output type: {type(output)}")
if isinstance(output, tuple):
    print(f"Output length: {len(output)}")
    decoded, raw = output
    print(f"  Decoded predictions: type {type(decoded)}, shape {decoded.shape if hasattr(decoded, 'shape') else 'N/A'}")
    print(f"  Raw predictions: type {type(raw)}")
    if isinstance(raw, list):
        print(f"    List length: {len(raw)}")
        for i, p in enumerate(raw):
            print(f"      Prediction {i}: shape {p.shape}")
elif isinstance(output, list):
    print(f"⚠️  WARNING: Eval mode returned list (should be tuple)")
    print(f"Output length: {len(output)}")
    for i, p in enumerate(output):
        print(f"  Prediction {i}: shape {p.shape}")
else:
    print(f"❌ ERROR: Unexpected output type: {type(output)}")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
if model.training:
    print("Model is in training mode - output should be list")
else:
    print("Model is in eval mode - output should be tuple (decoded, raw)")


