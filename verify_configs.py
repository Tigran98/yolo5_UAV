"""Verify fusion model config, data config, and dataloader"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from utils.fusion_dataloaders import DualModalDataset

print("="*60)
print("1. VERIFYING FUSION MODEL CONFIG")
print("="*60)
with open('models/yolov5n_fusion.yaml') as f:
    model_cfg = yaml.safe_load(f)
print(f"Fusion indices: {model_cfg['fusion_indices']}")
print(f"Fusion channels (YAML): {model_cfg['fusion_channels']}")
print(f"Width multiple: {model_cfg['width_multiple']}")
print(f"Expected actual channels: {[int(c * model_cfg['width_multiple']) for c in model_cfg['fusion_channels']]}")

print("\n" + "="*60)
print("2. VERIFYING DATA CONFIG")
print("="*60)
with open('data/anti_uav_fusion.yaml') as f:
    data_cfg = yaml.safe_load(f)
print(f"Path: {data_cfg['path']}")
print(f"Train: {data_cfg['train']}")
print(f"Val: {data_cfg['val']}")
print(f"Test: {data_cfg['test']}")
print(f"Classes: {data_cfg['nc']}")
print(f"Class names: {data_cfg['names']}")

print("\n" + "="*60)
print("3. VERIFYING DATALOADER LABEL HANDLING")
print("="*60)
try:
    # Try to create dataset
    train_path = Path(data_cfg['path']) / data_cfg['train']
    print(f"Train path: {train_path}")
    if train_path.exists():
        dataset = DualModalDataset(
            path=str(train_path),
            img_size=640,
            batch_size=1,
            augment=False,
            cache_images=False,
        )
        print(f"✓ Dataset created: {len(dataset)} image pairs")
        print(f"✓ Labels loaded: {len(dataset.labels)} label files")
        
        # Check a few labels
        if len(dataset.labels) > 0:
            sample_idx = 0
            labels = dataset.labels[sample_idx]
            print(f"\nSample label (index {sample_idx}):")
            print(f"  Shape: {labels.shape}")
            print(f"  Format: [class, x_center, y_center, width, height] (normalized)")
            if labels.size > 0:
                print(f"  First label: {labels[0]}")
                print(f"  Number of objects: {len(labels)}")
            else:
                print(f"  No objects in this image")
    else:
        print(f"✗ Train path does not exist: {train_path}")
except Exception as e:
    print(f"✗ Error creating dataset: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("VERIFICATION COMPLETE")
print("="*60)


