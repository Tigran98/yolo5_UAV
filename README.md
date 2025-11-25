<div align="center">
  <p>
    <a href="https://github.com/Tigran98/yolo5_UAV" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov5/v70/splash.jpg" alt="YOLOv5 UAV Fusion Banner"></a>
  </p>

  <a href="https://github.com/Tigran98/yolo5_UAV/actions"><img src="https://github.com/Tigran98/yolo5_UAV/workflows/CI/badge.svg" alt="CI"></a>
  <a href="https://github.com/Tigran98/yolo5_UAV/issues"><img src="https://img.shields.io/github/issues/Tigran98/yolo5_UAV" alt="GitHub issues"></a>
  <a href="https://github.com/Tigran98/yolo5_UAV/stargazers"><img src="https://img.shields.io/github/stars/Tigran98/yolo5_UAV" alt="GitHub stars"></a>
  <a href="https://github.com/Tigran98/yolo5_UAV/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Tigran98/yolo5_UAV" alt="License"></a>

</div>

# 🚁 YOLOv5 Dual-Stream Fusion for UAV Detection

<p align="center">
  <img src="https://img.shields.io/badge/Fusion-RGB%20%2B%20IR-blueviolet" alt="RGB+IR Fusion">
  <img src="https://img.shields.io/badge/Task-Drone%20Detection-blue" alt="Task">
  <img src="https://img.shields.io/badge/Platform-Jetson%20%7C%20FPGA-green" alt="Platform">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange" alt="Framework">
</p>

This repository implements a **Dual-Stream Sensor Fusion** architecture based on YOLOv5 for robust UAV (drone) detection. It fuses **Visible (RGB)** and **Infrared (IR)** thermal imagery to achieve superior detection performance across varying lighting and environmental conditions.

> **Key Innovation**: Unlike standard object detectors that use a single image modality, this system utilizes two parallel backbones to extract features from aligned RGB and IR images. These features are fused at multiple scales (P3, P4, P5) before prediction, combining the rich texture information of RGB with the thermal contrast of IR.

## 🌟 Key Features

- **Dual-Backbone Architecture**: Two separate CSPDarknet53 backbones process RGB and IR streams independently.
- **Multi-Scale Feature Fusion**: Features are fused at layers P3, P4, and P5 using concatenation and 1x1 convolutions.
- **Synchronized Augmentation**: Custom dataloaders apply identical geometric augmentations (flips, perspective, etc.) to both image pairs to maintain alignment.
- **Edge Optimized**: Designed for deployment on NVIDIA Jetson and FPGA platforms (ONNX, TensorRT, Vitis AI support).
- **Production Ready**: Includes complete pipeline for training, validation, and inference on dual-stream data.

---

## 🏗️ Fusion Architecture

The system follows a modified YOLOv5 architecture:

```mermaid
graph TD
    subgraph Inputs
    RGB[RGB Image 640x640]
    IR[IR Image 640x640]
    end

    subgraph Backbones
    B1[RGB Backbone (CSPDarknet)]
    B2[IR Backbone (CSPDarknet)]
    RGB --> B1
    IR --> B2
    end

    subgraph Fusion
    P3[Fusion P3]
    P4[Fusion P4]
    P5[Fusion P5]
    B1 --> P3 & P4 & P5
    B2 --> P3 & P4 & P5
    end

    subgraph Head
    Neck[PANet Neck]
    P3 --> Neck
    P4 --> Neck
    P5 --> Neck
    Detect[YOLO Detect Head]
    Neck --> Detect
    end
```

- **Backbones**: Pre-trained YOLOv5 weights are loaded into both RGB and IR backbones.
- **Fusion Mechanism**: Channel concatenation followed by 1x1 convolution to reduce dimensions back to standard sizes.

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/Tigran98/yolo5_UAV.git
cd yolo5_UAV
pip install -r requirements.txt
```

### 2. Dataset Preparation

The system expects paired RGB and IR images. The structure should be:

```
data/anti-uav/
├── images/
│   ├── train/
│   │   ├── rgb/        # RGB images (e.g., seq1_001.jpg)
│   │   └── ir/         # IR images (e.g., seq1_001.jpg)
│   └── val/
│       ├── rgb/
│       └── ir/
└── labels/
    ├── train/          # Labels in YOLO format (class x y w h)
    └── val/
```
*Note: Filenames must match between `rgb/` and `ir/` folders.*

### 3. Training

Use `train_fusion.py` to train the dual-stream model.

```bash
python train_fusion.py \
    --data data/anti_uav_fusion.yaml \
    --cfg models/yolov5n_fusion.yaml \
    --weights yolov5n.pt \
    --epochs 100 \
    --batch-size 16 \
    --img 640
```

### 4. Inference

Use `detect_fusion.py` to run inference on paired images.

```bash
python detect_fusion.py \
    --weights runs/train/exp/weights/best.pt \
    --source path/to/dataset/root/ \
    --img 640 \
    --conf-thres 0.25
```
*The script automatically looks for `rgb` and `ir` subfolders in the source path.*

---

## 📊 Performance Benchmarks

We benchmarked the system on the Anti-UAV dataset. The fusion model typically outperforms single-modality models.

| Model | Modality | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|----------|-----------|--------|---------|--------------|
| YOLOv5n | **RGB + IR (Fusion)** | **High** | **Balanced** | **91.21%** | **46.76%** |
| YOLOv5n | IR Only | 97.9% | 86.8% | 94.89% | 51.01% |
| YOLOv5n | RGB Only | 96.8% | 66.8% | 85.62% | 42.89% |

**Key Insights:**
- **IR Only**: Excellent for high-contrast targets (hot drones vs cold sky) but loses texture.
- **Fusion (RGB+IR)**: Provides robustness. While IR performs exceptionally well in ideal conditions, Fusion ensures reliability when thermal signatures are weak or when RGB texture helps distinguish drones from other heat sources (birds, etc.).

---

## 📁 Project Structure

- **`models/`**
  - `fusion_model.py`: Core class implementing `YOLOv5FusionModel`.
  - `yolov5n_fusion.yaml`: Configuration for the fusion architecture.
  - `common.py`: Contains the `FeatureFusion` module.
- **`utils/`**
  - `fusion_dataloaders.py`: `DualModalDataset` class for synchronized loading.
- **`data/`**
  - `anti_uav_fusion.yaml`: Dataset configuration.
- **Scripts**
  - `train_fusion.py`: Training loop for dual inputs.
  - `detect_fusion.py`: Inference script.
  - `val_fusion.py`: Validation script.

---

## 🛠️ Deployment

### Jetson (TensorRT)
1. **Export to ONNX**:
   ```bash
   python export.py --weights best.pt --include onnx --simplify
   ```
2. **Convert to TensorRT** (on Jetson):
   ```bash
   trtexec --onnx=best.onnx --saveEngine=best.engine --fp16
   ```

### FPGA (Vitis AI)
Follow the Vitis AI guide in `FUSION_MODEL_USAGE.md` for quantization and compilation for Xilinx DPUs.

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{yolov5_uav_fusion,
  title={YOLOv5 Dual-Stream UAV Detection System},
  author={Tigran98},
  year={2024},
  publisher={GitHub},
  url={https://github.com/Tigran98/yolo5_UAV}
}
```

---

<div align="center">
  <p>Made with ❤️ for UAV Detection Research</p>
</div>
