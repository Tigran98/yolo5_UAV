# YOLOv5 Fusion Model Usage Guide

## Overview
The fusion model combines RGB and IR (infrared) image modalities for improved object detection, specifically designed for drone detection tasks.

## Architecture
- **Dual Backbones**: Separate YOLOv5n backbones for RGB and IR streams
- **Fusion Points**: Features fused at P3, P4, P5 levels (after layers 4, 6, 9)
- **Fusion Method**: Concatenation + 1x1 convolution
- **Detection Head**: Standard YOLOv5 PANet neck and detection head

## Quick Start

### 1. Training

```bash
python train_fusion.py \
    --data data/anti_uav_fusion.yaml \
    --cfg models/yolov5n_fusion.yaml \
    --weights yolov5n.pt \
    --epochs 100 \
    --batch-size 16 \
    --img 640
```

**Key parameters:**
- `--data`: Dataset YAML with RGB-IR image pairs
- `--cfg`: Fusion model configuration
- `--weights`: Pre-trained YOLOv5 weights (will be loaded into both RGB and IR backbones)
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size (adjust based on GPU memory)
- `--img`: Input image size

### 2. Validation

```bash
python val_fusion.py \
    --weights runs/train/exp/weights/best.pt \
    --data data/anti_uav_fusion.yaml \
    --img 640 \
    --batch-size 32
```

### 3. Inference

```bash
python detect_fusion.py \
    --weights runs/train/exp/weights/best.pt \
    --source path/to/rgb_images/ \
    --img 640 \
    --conf-thres 0.25
```

**Note**: IR images should be in the same directory with `infrared` replacing `visible` in filenames.
- RGB: `image_visible_001.jpg`
- IR: `image_infrared_001.jpg`

## Dataset Structure

```
anti-uav/
├── images/
│   ├── train/
│   │   ├── image_visible_001.jpg
│   │   ├── image_infrared_001.jpg
│   │   └── ...
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    │   ├── image_visible_001.txt
    │   └── ...
    └── val/
```

## Model Configuration

The fusion model configuration (`models/yolov5n_fusion.yaml`) defines:
- Fusion points: `[4, 6, 9]` (P3, P4, P5)
- Fusion channels: `[64, 128, 256]` (before width_multiple)
- Number of classes: `1` (drone)

## Key Features

1. **Synchronized Augmentation**: RGB and IR images undergo the same geometric transforms
2. **Weight Transfer**: Pre-trained YOLOv5 weights are loaded into both backbones
3. **Multi-scale Fusion**: Features fused at three different scales for robust detection

## Implementation Files

- `models/fusion_model.py`: Main fusion model class
- `models/common.py`: FeatureFusion module
- `utils/fusion_dataloaders.py`: Dual-modal dataset loader
- `train_fusion.py`: Training script
- `val_fusion.py`: Validation script
- `detect_fusion.py`: Inference script

## Troubleshooting

### Issue: "No IR pair found for [image]"
- Ensure IR images have `infrared` in filename (replacing `visible`)
- Check that RGB and IR images are in the same directory

### Issue: Channel mismatch errors
- This should be automatically handled, but if it occurs, check fusion channel configuration
- Ensure `fusion_channels` in YAML match actual backbone output channels

### Issue: Model initialization fails
- Verify YAML configuration is correct
- Check that all required modules are imported
- Ensure PyTorch version is compatible (>=1.7.0)

## Performance Notes

- Model has ~2.98M parameters (dual backbones)
- Training requires approximately 2x memory compared to single-stream YOLOv5n
- Inference time is ~1.5-2x slower due to dual inputs

## Next Steps

1. Train on your dataset
2. Evaluate with `val_fusion.py`
3. Run inference on test images
4. Fine-tune hyperparameters as needed


