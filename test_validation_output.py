"""Debug validation output unpacking"""
import torch
from models.fusion_model import YOLOv5FusionModel

model = YOLOv5FusionModel(nc=1)
model.eval()  # Validation mode

x_rgb = torch.randn(2, 3, 640, 640)
x_ir = torch.randn(2, 3, 640, 640)

print("=== Test compute_loss=True branch ===")
compute_loss = True
if compute_loss:
    out = model(x_rgb, x_ir)
    print(f"model(rgb, ir) returns: {type(out)}")
    if isinstance(out, tuple):
        print(f"  Tuple length: {len(out)}")
        print(f"  Item 0 type: {type(out[0])}, shape: {out[0].shape if isinstance(out[0], torch.Tensor) else 'N/A'}")
        print(f"  Item 1 type: {type(out[1])}")
    
    # Simulate unpacking
    preds, train_out = out
    print(f"\nAfter unpacking:")
    print(f"  preds type: {type(preds)}, shape: {preds.shape}")
    print(f"  train_out type: {type(train_out)}")
    if isinstance(train_out, list):
        print(f"  train_out is list with {len(train_out)} items")
        for i, t in enumerate(train_out):
            if isinstance(t, torch.Tensor):
                print(f"    Item {i}: {t.shape}")

print("\n=== Test compute_loss=False branch ===")
compute_loss = False
if not compute_loss:
    out = (model(x_rgb, x_ir, augment=False), None)
    print(f"(model(...), None) creates: {type(out)}")
    print(f"  Tuple length: {len(out)}")
    print(f"  Item 0 type: {type(out[0])}")
    if isinstance(out[0], tuple):
        print(f"    Item 0 is tuple with length: {len(out[0])}")
        print(f"      Sub-item 0 type: {type(out[0][0])}")
        if isinstance(out[0][0], torch.Tensor):
            print(f"      Sub-item 0 shape: {out[0][0].shape}")
    elif isinstance(out[0], torch.Tensor):
        print(f"    Item 0 shape: {out[0].shape}")
    print(f"  Item 1: {out[1]}")
    
    # Simulate unpacking
    preds, train_out = out
    print(f"\nAfter unpacking:")
    print(f"  preds type: {type(preds)}")
    if isinstance(preds, tuple):
        print(f"  preds is TUPLE (WRONG!) with {len(preds)} items")
        print(f"    This will break NMS!")
    elif isinstance(preds, torch.Tensor):
        print(f"  preds shape: {preds.shape} (CORRECT)")
    print(f"  train_out: {train_out}")


