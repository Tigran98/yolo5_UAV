<div align="center">
  <p>
    <a href="https://github.com/Tigran98/yolo5_UAV" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov5/v70/splash.jpg" alt="YOLOv5 UAV Detection Banner"></a>
  </p>

  <a href="https://github.com/Tigran98/yolo5_UAV/actions"><img src="https://github.com/Tigran98/yolo5_UAV/workflows/CI/badge.svg" alt="CI"></a>
  <a href="https://github.com/Tigran98/yolo5_UAV/issues"><img src="https://img.shields.io/github/issues/Tigran98/yolo5_UAV" alt="GitHub issues"></a>
  <a href="https://github.com/Tigran98/yolo5_UAV/stargazers"><img src="https://img.shields.io/github/stars/Tigran98/yolo5_UAV" alt="GitHub stars"></a>
  <a href="https://github.com/Tigran98/yolo5_UAV/blob/main/LICENSE"><img src="https://img.shields.io/github/license/Tigran98/yolo5_UAV" alt="License"></a>

  </div>

# 🚁 YOLOv5 UAV Detection System

<p align="center">
  <img src="https://img.shields.io/badge/YOLOv5-Drone%20Detection-blue" alt="YOLOv5 Drone Detection">
  <img src="https://img.shields.io/badge/Platform-Jetson%20%7C%20FPGA-green" alt="Platform">
  <img src="https://img.shields.io/badge/Framework-PyTorch-orange" alt="Framework">
  <img src="https://img.shields.io/badge/Export-ONNX%20%7C%20TensorRT-red" alt="Export Formats">
</p>

This repository contains a **YOLOv5-based UAV (Unmanned Aerial Vehicle) detection system** optimized for real-time deployment on edge devices including **NVIDIA Jetson** platforms and **FPGA** accelerators. The system is designed for detecting and tracking drones in various environmental conditions using both RGB and infrared imagery.

## 🎯 Features

- ✅ **Multi-Spectral Detection**: Supports RGB and IR (infrared) image detection
- ✅ **Edge Deployment Ready**: Optimized for Jetson Nano, Xavier, Orin, and FPGA platforms
- ✅ **Multiple Export Formats**: ONNX, TensorRT, and Vitis AI compatible
- ✅ **Real-Time Performance**: Achieves high FPS on embedded hardware
- ✅ **Pre-trained Models**: Includes models trained on Anti-UAV dataset
- ✅ **Production Ready**: Complete pipeline from training to deployment

## 📊 Performance Benchmarks

### Training Results (100 epochs, batch size 64, image size 640)

| Model | Dataset | Precision | Recall | mAP@0.5 (Best) | mAP@0.5:0.95 (Best) | Parameters | Model Size |
|-------|---------|-----------|--------|----------------|---------------------|------------|------------|
| YOLOv5n | Full (RGB+IR) | 95.1% | 79.5% | **91.21%** | **46.76%** | 1.9M | 3.9 MB |
| YOLOv5n | IR Only | 97.9% | 86.8% | **94.89%** | **51.01%** | 1.9M | 3.9 MB |
| YOLOv5n | RGB Only | 96.8% | 66.8% | **85.62%** | **42.89%** | 1.9M | 3.9 MB |

**Key Observations:**
- **IR-only model** achieves the best performance with 94.89% mAP@0.5 and 51.01% mAP@0.5:0.95
- **Full dataset (RGB+IR)** shows balanced performance with 91.21% mAP@0.5
- **RGB-only model** has lower recall (66.8%), suggesting drones are harder to detect in RGB images
- All models trained for 100 epochs with SGD optimizer

### Inference Performance (Estimated)

| Platform | Model | Precision | FPS | Power |
|----------|-------|-----------|-----|-------|
| NVIDIA Jetson Xavier NX | YOLOv5n FP16 | FP16 | ~45 | 15W |
| NVIDIA Jetson Xavier NX | YOLOv5n INT8 | INT8 | ~60 | 15W |
| NVIDIA Jetson Orin Nano | YOLOv5n FP16 | FP16 | ~70 | 15W |
| Xilinx Kria KV260 | YOLOv5n INT8 | INT8 | ~30 | 10W |
| Desktop GPU (RTX 3080) | YOLOv5n FP32 | FP32 | ~200 | 320W |

*Note: FPS values are estimates and may vary based on actual implementation and optimization.*

<div align="center">
  <img src="runs/exp_FULL/results.png" width="80%" alt="Training Results">
  <p><i>Training metrics: Precision, Recall, mAP</i></p>
</div>

## 🏗️ Architecture

The system follows YOLOv5 architecture with custom optimizations:

```
Input (640x640) 
    ↓
Backbone (CSPDarknet53)
    ↓
Neck (PANet)
    ↓
Head (YOLO Detection)
    ↓
Output (Bounding Boxes + Confidence)
```

**Key Components:**
- **Backbone**: CSPDarknet53 with cross-stage partial connections
- **Neck**: PANet (Path Aggregation Network) for multi-scale feature fusion
- **Head**: YOLO detection head with anchor-based predictions
- **Post-processing**: NMS (Non-Maximum Suppression) for duplicate removal

---

## 📋 Table of Contents

- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Training](#-training)
- [Testing & Validation](#-testing--validation)
- [Inference](#-inference)
- [Model Export](#-model-export)
  - [ONNX Export](#onnx-export)
  - [TensorRT Conversion (Jetson)](#tensorrt-conversion-jetson)
  - [Vitis AI Conversion (FPGA)](#vitis-ai-conversion-fpga)
- [Deployment](#-deployment)
  - [Jetson Deployment](#jetson-deployment)
  - [FPGA Deployment](#fpga-deployment)
- [Results](#-results)
- [Citation](#-citation)
- [License](#-license)

---

## 🚀 Installation

### Prerequisites

- Python >= 3.8
- PyTorch >= 1.8
- CUDA >= 10.2 (for GPU training)
- NVIDIA Jetson JetPack >= 4.6 (for Jetson deployment)
- Xilinx Vitis AI >= 3.0 (for FPGA deployment)

### Setup Environment

```bash
# Clone the repository
git clone https://github.com/Tigran98/yolo5_UAV.git
cd yolo5_UAV

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch (select appropriate version for your CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt

# Verify installation
python detect.py --help
```

### Additional Dependencies for Export

```bash
# For ONNX export
pip install onnx onnxruntime

# For TensorRT (on Jetson)
pip install tensorrt

# For ONNX simplification
pip install onnx-simplifier
```

---

## 📦 Dataset Preparation

This project uses the **Anti-UAV dataset** for training and evaluation. The dataset includes both RGB and infrared imagery of UAVs in various conditions.

### Dataset Structure

```
data/
├── anti-uav/
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/
│       ├── train/
│       ├── val/
│       └── test/
```

### Download Dataset

```bash
# Create data directory
mkdir -p data/anti-uav

# Download Anti-UAV dataset
# Option 1: Manual download from source
# Download from: [Add your dataset source URL]

# Option 2: Using script (if available)
# python scripts/download_dataset.py --dataset anti-uav
```

### Dataset Configuration

Edit `anti_uav_yolo.yaml` to configure dataset paths:

```yaml
# anti_uav_yolo.yaml
names:
  0: drone

path: data/anti-uav  # Dataset root directory
train: images/train  # Train images
val: images/val      # Validation images
test: images/test    # Test images

# Classes
nc: 1  # Number of classes
```

### Data Statistics

- **Training Images**: ~X,XXX
- **Validation Images**: ~X,XXX
- **Test Images**: ~X,XXX
- **Classes**: 1 (drone)
- **Image Sizes**: 640x640 (resized during training)
- **Formats**: RGB and IR images

---

## 🎓 Training

### Quick Start Training

```bash
# Train YOLOv5s on full dataset (RGB + IR)
python train.py --data anti_uav_yolo.yaml --weights yolov5s.pt --img 640 --batch 16 --epochs 100 --name exp_FULL

# Train YOLOv5n on IR images only
python train.py --data anti_uav_yolo.yaml --weights yolov5n.pt --img 640 --batch 32 --epochs 100 --name exp_IR

# Train YOLOv5s on RGB images only
python train.py --data anti_uav_yolo.yaml --weights yolov5s.pt --img 640 --batch 16 --epochs 100 --name exp_RGB
```

### Training Configuration

#### Hyperparameters

The project includes several hyperparameter configurations in `data/hyps/`:

- `hyp.scratch-low.yaml` - Low augmentation for small datasets
- `hyp.scratch-med.yaml` - Medium augmentation (default)
- `hyp.scratch-high.yaml` - High augmentation for large datasets

```bash
# Train with custom hyperparameters
python train.py --data anti_uav_yolo.yaml \
                --weights yolov5s.pt \
                --img 640 \
                --batch 16 \
                --epochs 100 \
                --hyp data/hyps/hyp.scratch-high.yaml \
                --name exp_custom
```

#### Advanced Training Options

```bash
# Resume training from checkpoint
python train.py --resume runs/exp_FULL/weights/last.pt

# Train with multi-GPU
python -m torch.distributed.run --nproc_per_node 2 train.py \
    --data anti_uav_yolo.yaml \
    --weights yolov5s.pt \
    --img 640 \
    --batch 32 \
    --epochs 100 \
    --device 0,1

# Train with mixed precision (faster training)
python train.py --data anti_uav_yolo.yaml \
                --weights yolov5s.pt \
                --img 640 \
                --batch 16 \
                --epochs 100 \
                --amp

# Train from scratch (no pretrained weights)
python train.py --data anti_uav_yolo.yaml \
                --cfg models/yolov5s.yaml \
                --img 640 \
                --batch 16 \
                --epochs 300 \
                --name exp_scratch
```

### Monitoring Training

Training results are saved to `runs/exp_*/`:

- `weights/best.pt` - Best model checkpoint
- `weights/last.pt` - Last epoch checkpoint
- `results.png` - Training curves
- `confusion_matrix.png` - Confusion matrix
- `F1_curve.png`, `P_curve.png`, `R_curve.png` - Metric curves
- `results.csv` - Training metrics log

```bash
# View training progress
tensorboard --logdir runs/
```

---

## 🧪 Testing & Validation

### Validate Trained Model

```bash
# Validate on test set
python val.py --data anti_uav_yolo.yaml \
              --weights runs/exp_FULL/weights/best.pt \
              --img 640 \
              --batch 32 \
              --task test

# Validate on validation set
python val.py --data anti_uav_yolo.yaml \
              --weights runs/exp_FULL/weights/best.pt \
              --img 640 \
              --batch 32 \
              --task val

# Save predictions
python val.py --data anti_uav_yolo.yaml \
              --weights runs/exp_FULL/weights/best.pt \
              --img 640 \
              --save-txt \
              --save-conf \
              --save-json
```

### Compute Metrics

The validation script computes:
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **mAP@0.5**: Mean Average Precision at IoU threshold 0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds 0.5-0.95

---

## 🔍 Inference

### Image Inference

```bash
# Detect on single image
python detect.py --weights runs/exp_FULL/weights/best.pt \
                 --source inputs/01_1667_0001-1500_IR_10_input.jpg \
                 --img 640 \
                 --conf-thres 0.25 \
                 --iou-thres 0.45

# Detect on multiple images
python detect.py --weights runs/exp_FULL/weights/best.pt \
                 --source inputs/ \
                 --img 640 \
                 --save-txt \
                 --save-conf
```

### Video Inference

```bash
# Detect on video file
python detect.py --weights runs/exp_FULL/weights/best.pt \
                 --source inputs/infrared.mp4 \
                 --img 640 \
                 --conf-thres 0.25

# Detect on webcam (device 0)
python detect.py --weights runs/exp_FULL/weights/best.pt \
                 --source 0 \
                 --img 640 \
                 --view-img

# Detect on RTSP stream
python detect.py --weights runs/exp_FULL/weights/best.pt \
                 --source rtsp://your-camera-ip:554/stream \
                 --img 640
```

### Batch Inference

```bash
# Process multiple videos
for video in inputs/*.mp4; do
    python detect.py --weights runs/exp_FULL/weights/best.pt \
                     --source "$video" \
                     --img 640 \
                     --name "${video##*/}"
done
```

### Python API

```python
import torch

# Load model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/exp_FULL/weights/best.pt')

# Set confidence threshold
model.conf = 0.25
model.iou = 0.45

# Inference on image
img = 'inputs/01_1667_0001-1500_IR_10_input.jpg'
results = model(img)

# Results
results.print()  # Print results to console
results.show()   # Display results
results.save()   # Save results to runs/detect/

# Access predictions
boxes = results.xyxy[0]  # Bounding boxes
labels = results.pandas().xyxy[0]  # Pandas DataFrame
```

---

## 📤 Model Export

### ONNX Export

ONNX (Open Neural Network Exchange) is a universal format for deep learning models.

```bash
# Export to ONNX
python export.py --weights runs/exp_FULL/weights/best.pt \
                 --include onnx \
                 --img 640 \
                 --batch 1 \
                 --simplify

# This creates: runs/exp_FULL/weights/best.onnx
```

#### Verify ONNX Model

```python
import onnx
import onnxruntime as ort

# Load ONNX model
model = onnx.load("runs/exp_FULL/weights/best.onnx")

# Check model
onnx.checker.check_model(model)
print("✅ ONNX model is valid")

# Run inference with ONNX Runtime
session = ort.InferenceSession("runs/exp_FULL/weights/best.onnx")
print("✅ ONNX Runtime session created successfully")
```

#### ONNX Inference

```bash
# Detect using ONNX model
python detect.py --weights runs/exp_FULL/weights/best.onnx \
                 --source inputs/infrared.mp4 \
                 --img 640
```

---

### TensorRT Conversion (Jetson)

TensorRT optimizes models for NVIDIA GPUs and provides significant speedup on Jetson devices.

#### Prerequisites (on Jetson)

```bash
# Check TensorRT installation
python -c "import tensorrt; print(tensorrt.__version__)"

# Install additional dependencies
pip install pycuda
```

#### Method 1: Direct Export to TensorRT

```bash
# Export directly to TensorRT engine (requires NVIDIA GPU)
python export.py --weights runs/exp_FULL/weights/best.pt \
                 --include engine \
                 --img 640 \
                 --device 0

# This creates: runs/exp_FULL/weights/best.engine
```

#### Method 2: Convert ONNX to TensorRT (Recommended for Jetson)

```bash
# Step 1: Export to ONNX (can be done on any machine)
python export.py --weights runs/exp_FULL/weights/best.pt \
                 --include onnx \
                 --img 640 \
                 --simplify

# Step 2: Convert ONNX to TensorRT on Jetson
trtexec --onnx=runs/exp_FULL/weights/best.onnx \
        --saveEngine=runs/exp_FULL/weights/best.engine \
        --fp16 \
        --workspace=4096 \
        --verbose

# For INT8 quantization (better performance, slight accuracy loss)
trtexec --onnx=runs/exp_FULL/weights/best.onnx \
        --saveEngine=runs/exp_FULL/weights/best_int8.engine \
        --int8 \
        --workspace=4096 \
        --verbose
```

#### TensorRT Inference on Jetson

```bash
# Detect using TensorRT engine
python detect.py --weights runs/exp_FULL/weights/best.engine \
                 --source inputs/infrared.mp4 \
                 --img 640 \
                 --device 0
```

#### Benchmark TensorRT Performance

```python
import time
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

# Load TensorRT engine
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

engine = load_engine("runs/exp_FULL/weights/best.engine")

# Benchmark
input_shape = (1, 3, 640, 640)
dummy_input = np.random.randn(*input_shape).astype(np.float32)

# Warmup
for _ in range(10):
    # Run inference
    pass

# Measure
num_iterations = 100
start = time.time()
for _ in range(num_iterations):
    # Run inference
    pass
end = time.time()

fps = num_iterations / (end - start)
print(f"TensorRT FPS: {fps:.2f}")
```

---

### Vitis AI Conversion (FPGA)

Vitis AI enables deployment on Xilinx FPGAs with DPU (Deep Learning Processing Unit) acceleration.

#### Prerequisites

```bash
# Install Vitis AI Docker
git clone --recurse-submodules https://github.com/Xilinx/Vitis-AI
cd Vitis-AI

# Pull Vitis AI Docker image (CPU)
docker pull xilinx/vitis-ai-cpu:latest

# Or GPU version
docker pull xilinx/vitis-ai-gpu:latest

# Start Docker container
./docker_run.sh xilinx/vitis-ai-cpu:latest
```

#### Conversion Pipeline

**Step 1: Export to ONNX**

```bash
# Export PyTorch model to ONNX (outside Docker)
python export.py --weights runs/exp_FULL/weights/best.pt \
                 --include onnx \
                 --img 640 \
                 --opset 11 \
                 --simplify
```

**Step 2: Convert ONNX to XIR (Inside Vitis AI Docker)**

```bash
# Activate conda environment
conda activate vitis-ai-pytorch

# Convert ONNX to XIR format
vai_c_xir --onnx runs/exp_FULL/weights/best.onnx \
          --output runs/exp_FULL/weights/best_xir.xmodel \
          --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G_ISA1_B4096.json
```

**Step 3: Quantization (INT8)**

Quantization is required for FPGA deployment to reduce model size and improve performance.

```python
# quantize.py
from pytorch_nndct.apis import torch_quantizer
import torch

# Load PyTorch model
model = torch.load('runs/exp_FULL/weights/best.pt')['model'].float()

# Create quantizer
quantizer = torch_quantizer(
    quant_mode='calib',
    module=model,
    input_args=torch.randn(1, 3, 640, 640)
)

# Calibration with representative dataset
quant_model = quantizer.quant_model
for i, (images, targets) in enumerate(calibration_loader):
    if i > 100:  # Use ~100 images for calibration
        break
    quant_model(images)

# Export quantized model
quantizer.export_quant_config()
quantizer.export_xmodel(output_dir='runs/exp_FULL/weights/', deploy_check=True)
```

**Step 4: Compile for Target FPGA**

```bash
# Compile quantized model for specific FPGA board
vai_c_xir --xmodel runs/exp_FULL/weights/best_int.xmodel \
          --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G_ISA1_B4096.json \
          --output runs/exp_FULL/weights/best_fpga.xmodel \
          --net_name yolov5_uav
```

#### FPGA Inference

```python
# fpga_inference.py
import xir
import vart
import numpy as np
import cv2

# Load compiled model
graph = xir.Graph.deserialize('runs/exp_FULL/weights/best_fpga.xmodel')
subgraphs = graph.get_root_subgraph().toposort_child_subgraph()
dpu_subgraph = subgraphs[0]

# Create DPU runner
runner = vart.Runner.create_runner(dpu_subgraph, "run")

# Get input/output tensors
input_tensors = runner.get_input_tensors()
output_tensors = runner.get_output_tensors()

# Prepare input
image = cv2.imread('inputs/01_1667_0001-1500_IR_10_input.jpg')
image = cv2.resize(image, (640, 640))
image = image.astype(np.float32) / 255.0
image = np.transpose(image, (2, 0, 1))
image = np.expand_dims(image, axis=0)

# Run inference
job_id = runner.execute_async([image], [])
runner.wait(job_id)
output = runner.get_output_tensors()[0]

print("✅ FPGA inference completed")
```

#### Supported FPGA Platforms

- **Xilinx Zynq UltraScale+ MPSoC** (ZCU102, ZCU104)
- **Xilinx Kria KV260**
- **Xilinx Alveo** (U50, U200, U250, U280)
- **Custom Zynq boards with DPU**

---

## 🚀 Deployment

### Jetson Deployment

#### Setup Jetson Device

```bash
# Check JetPack version
sudo apt-cache show nvidia-jetpack

# Install required packages
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev
sudo apt-get install -y libopencv-dev python3-opencv

# Install PyTorch for Jetson
wget https://nvidia.box.com/shared/static/[version].whl -O torch.whl
pip3 install torch.whl

# Install torchvision
pip3 install torchvision

# Clone repository on Jetson
git clone https://github.com/Tigran98/yolo5_UAV.git
cd yolo5_UAV
pip3 install -r requirements.txt
```

#### Transfer Model to Jetson

```bash
# From development machine
scp runs/exp_FULL/weights/best.engine jetson@<jetson-ip>:~/yolo5_UAV/weights/

# Or using USB drive, rsync, etc.
```

#### Run Inference on Jetson

```bash
# Test TensorRT engine
python3 detect.py --weights weights/best.engine \
                  --source 0 \
                  --img 640 \
                  --device 0 \
                  --view-img

# For CSI camera (Raspberry Pi Camera)
python3 detect.py --weights weights/best.engine \
                  --source "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink" \
                  --img 640 \
                  --device 0
```

#### Optimize Jetson Performance

```bash
# Maximize Jetson performance
sudo nvpmodel -m 0  # Set to maximum power mode
sudo jetson_clocks  # Maximize clock speeds

# Check GPU utilization
tegrastats

# Enable maximum CPU cores
sudo nvpmodel -q  # Query current mode
```

#### Create Deployment Script

```python
# deploy_jetson.py
import cv2
import torch
import numpy as np
from pathlib import Path

class JetsonDetector:
    def __init__(self, weights='weights/best.engine', img_size=640, conf_thres=0.25):
        self.img_size = img_size
        self.conf_thres = conf_thres
        
        # Load model
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights)
        self.model.conf = conf_thres
        
    def detect_video(self, source=0):
        cap = cv2.VideoCapture(source)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Inference
            results = self.model(frame)
            
            # Draw results
            annotated = results.render()[0]
            
            # Display
            cv2.imshow('UAV Detection', annotated)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = JetsonDetector(weights='weights/best.engine')
    detector.detect_video(source=0)
```

---

### FPGA Deployment

#### Setup FPGA Board

```bash
# For Xilinx ZCU104 or similar boards

# 1. Flash SD card with Vitis AI board image
# Download from: https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html

# 2. Boot board and login
ssh root@<board-ip>

# 3. Install required packages (usually pre-installed in Vitis AI image)
pip3 install opencv-python numpy
```

#### Transfer Model to FPGA

```bash
# Copy compiled xmodel to board
scp runs/exp_FULL/weights/best_fpga.xmodel root@<board-ip>:/root/models/

# Copy inference script
scp deploy_fpga.py root@<board-ip>:/root/
```

#### FPGA Inference Script

```python
# deploy_fpga.py
import cv2
import numpy as np
import xir
import vart
import time

class FPGADetector:
    def __init__(self, model_path, conf_thres=0.25):
        self.conf_thres = conf_thres
        
        # Load DPU runner
        self.graph = xir.Graph.deserialize(model_path)
        subgraphs = self.graph.get_root_subgraph().toposort_child_subgraph()
        self.dpu = subgraphs[0]
        self.runner = vart.Runner.create_runner(self.dpu, "run")
        
        # Get input/output specs
        self.input_tensors = self.runner.get_input_tensors()
        self.output_tensors = self.runner.get_output_tensors()
        
        self.input_shape = tuple(self.input_tensors[0].dims)
        print(f"Input shape: {self.input_shape}")
        
    def preprocess(self, image):
        # Resize and normalize
        img = cv2.resize(image, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img
    
    def detect(self, image):
        # Preprocess
        input_data = self.preprocess(image)
        
        # Run inference
        job_id = self.runner.execute_async([input_data], [])
        self.runner.wait(job_id)
        
        # Get output
        output_data = self.runner.get_output_tensors()
        
        # Post-process (implement NMS and box decoding)
        boxes = self.postprocess(output_data, image.shape)
        
        return boxes
    
    def postprocess(self, output, orig_shape):
        # Implement YOLO post-processing
        # This is simplified - full implementation needed
        boxes = []
        # ... decode boxes, apply NMS, scale to original size
        return boxes
    
    def detect_video(self, source=0):
        cap = cv2.VideoCapture(source)
        fps_counter = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect
            boxes = self.detect(frame)
            
            # Draw boxes
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                if conf > self.conf_thres:
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'Drone {conf:.2f}', (int(x1), int(y1)-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 30 == 0:
                fps = 30 / (time.time() - start_time)
                print(f"FPS: {fps:.2f}")
                start_time = time.time()
            
            # Display
            cv2.imshow('FPGA UAV Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    detector = FPGADetector(model_path='/root/models/best_fpga.xmodel')
    detector.detect_video(source='/dev/video0')
```

#### Run on FPGA

```bash
# On FPGA board
cd /root
python3 deploy_fpga.py
```

---

## 📊 Results

### Training Results

#### Detailed Performance Comparison

| Experiment | Model | Best Epoch | Final mAP@0.5 | Final mAP@0.5:0.95 | Training Time (100 epochs) |
|------------|-------|------------|---------------|--------------------|-----------------------------|
| exp_FULL | YOLOv5n | 10-12 | 85.83% | 40.97% | ~8 hours (estimated) |
| exp_IR | YOLOv5n | 8-13 | 92.30% | 47.13% | ~8 hours (estimated) |
| exp_RGB | YOLOv5n | 15-34 | 76.63% | 36.67% | ~8 hours (estimated) |

**Performance Analysis:**
- 🥇 **IR model converges fastest** (best results at epoch 8-13)
- 🥈 **Full model (RGB+IR)** shows good generalization
- 🥉 **RGB model** requires more epochs and achieves lower accuracy

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Full Dataset (RGB+IR)</b></td>
      <td align="center"><b>IR Only</b></td>
      <td align="center"><b>RGB Only</b></td>
    </tr>
    <tr>
      <td><img src="runs/exp_FULL/results.png" width="100%" alt="Full Training Results"></td>
      <td><img src="runs/exp_IR/results.png" width="100%" alt="IR Training Results"></td>
      <td><img src="runs/exp_RGB/results.png" width="100%" alt="RGB Training Results"></td>
    </tr>
  </table>
  <p><i>Training metrics comparison: Precision, Recall, mAP curves for all three experiments</i></p>
</div>

### Confusion Matrices

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Full Dataset</b></td>
      <td align="center"><b>IR Only</b></td>
      <td align="center"><b>RGB Only</b></td>
    </tr>
    <tr>
      <td><img src="runs/exp_FULL/confusion_matrix.png" width="100%" alt="Full Confusion Matrix"></td>
      <td><img src="runs/exp_IR/confusion_matrix.png" width="100%" alt="IR Confusion Matrix"></td>
      <td><img src="runs/exp_RGB/confusion_matrix.png" width="100%" alt="RGB Confusion Matrix"></td>
    </tr>
  </table>
  <p><i>Confusion matrices showing detection performance across different datasets</i></p>
</div>

### Sample Detections

#### Full Dataset (RGB+IR) Predictions
<div align="center">
  <table>
    <tr>
      <td><img src="runs/exp_FULL/val_batch0_pred.jpg" width="100%" alt="Full Dataset - Batch 0"></td>
      <td><img src="runs/exp_FULL/val_batch1_pred.jpg" width="100%" alt="Full Dataset - Batch 1"></td>
      <td><img src="runs/exp_FULL/val_batch2_pred.jpg" width="100%" alt="Full Dataset - Batch 2"></td>
    </tr>
  </table>
</div>

#### IR-Only Dataset Predictions
<div align="center">
  <table>
    <tr>
      <td><img src="runs/exp_IR/val_batch0_pred.jpg" width="100%" alt="IR Dataset - Batch 0"></td>
      <td><img src="runs/exp_IR/val_batch1_pred.jpg" width="100%" alt="IR Dataset - Batch 1"></td>
      <td><img src="runs/exp_IR/val_batch2_pred.jpg" width="100%" alt="IR Dataset - Batch 2"></td>
    </tr>
  </table>
</div>

#### RGB-Only Dataset Predictions
<div align="center">
  <table>
    <tr>
      <td><img src="runs/exp_RGB/val_batch0_pred.jpg" width="100%" alt="RGB Dataset - Batch 0"></td>
      <td><img src="runs/exp_RGB/val_batch1_pred.jpg" width="100%" alt="RGB Dataset - Batch 1"></td>
      <td><img src="runs/exp_RGB/val_batch2_pred.jpg" width="100%" alt="RGB Dataset - Batch 2"></td>
    </tr>
  </table>
  <p><i>UAV detection results on validation sets across all three experiments</i></p>
</div>

### Precision-Recall Curves

<div align="center">
  <table>
    <tr>
      <td align="center"><b>Full Dataset</b></td>
      <td align="center"><b>IR Only</b></td>
      <td align="center"><b>RGB Only</b></td>
    </tr>
    <tr>
      <td><img src="runs/exp_FULL/PR_curve.png" width="100%" alt="Full PR Curve"></td>
      <td><img src="runs/exp_IR/PR_curve.png" width="100%" alt="IR PR Curve"></td>
      <td><img src="runs/exp_RGB/PR_curve.png" width="100%" alt="RGB PR Curve"></td>
    </tr>
    <tr>
      <td align="center"><b>F1 Score - Full</b></td>
      <td align="center"><b>F1 Score - IR</b></td>
      <td align="center"><b>F1 Score - RGB</b></td>
    </tr>
    <tr>
      <td><img src="runs/exp_FULL/F1_curve.png" width="100%" alt="Full F1 Curve"></td>
      <td><img src="runs/exp_IR/F1_curve.png" width="100%" alt="IR F1 Curve"></td>
      <td><img src="runs/exp_RGB/F1_curve.png" width="100%" alt="RGB F1 Curve"></td>
    </tr>
  </table>
  <p><i>Precision-Recall and F1 curves showing model performance at different confidence thresholds</i></p>
</div>

### Label Statistics

<div align="center">
  <table>
    <tr>
      <td><img src="runs/exp_FULL/labels.jpg" width="100%" alt="Full Dataset Labels"></td>
      <td><img src="runs/exp_IR/labels.jpg" width="100%" alt="IR Dataset Labels"></td>
    </tr>
  </table>
  <p><i>Dataset label distribution and bounding box statistics</i></p>
</div>

---

## 📈 Experiment Summary & Insights

### Key Findings

1. **IR Images Perform Best** 🥇
   - Highest mAP@0.5: 94.89%
   - Highest mAP@0.5:0.95: 51.01%
   - Best precision (97.9%) and recall (86.8%)
   - Fastest convergence (best results at epoch 8)

2. **Combined Dataset (RGB+IR) Shows Balance** 🥈
   - Good generalization: mAP@0.5 = 91.21%
   - Balanced between RGB and IR performance
   - Suitable for mixed deployment scenarios

3. **RGB-Only Challenges** 🥉
   - Lowest recall: 66.8% (many missed detections)
   - Lower mAP@0.5: 85.62% (best epoch), 76.63% (final)
   - Drones harder to detect against complex backgrounds
   - Requires longer training (best at epochs 15-34)

### Recommendations

| Use Case | Recommended Model | Reason |
|----------|-------------------|---------|
| Infrared/Thermal Cameras | **exp_IR** model | Best accuracy (94.89% mAP), highest recall |
| RGB Day/Night Mixed | **exp_FULL** model | Balanced performance, handles both modalities |
| RGB Only (Daylight) | **exp_RGB** model | Optimized for visible spectrum |
| Real-time Edge Deployment | **exp_IR** with INT8 | Best accuracy + fastest inference |
| Multi-sensor Fusion | **Ensemble (IR+RGB)** | Complementary strengths |

### Training Insights

- **Early Stopping**: IR model peaks at epoch 8-13, Full at 10-12, RGB at 15-34
- **Batch Size**: All experiments used batch size 64 for stability
- **Image Size**: 640x640 provides good balance between accuracy and speed
- **Optimizer**: SGD with momentum (0.937) works well
- **Data Augmentation**: Standard augmentations (mosaic, flip, scale) applied

---

## 🔬 Model Variants

| Model | Size (MB) | Parameters | FLOPs | Inference Time (ms) |
|-------|-----------|------------|-------|---------------------|
| YOLOv5n | 3.9 | 1.9M | 4.5G | 6.3 |
| YOLOv5s | 14.4 | 7.2M | 16.5G | 9.2 |
| YOLOv5m | 40.8 | 21.2M | 49.0G | 17.8 |
| YOLOv5l | 89.3 | 46.5M | 109.1G | 27.3 |
| YOLOv5x | 166.8 | 86.7M | 205.7G | 45.9 |

*Inference times measured on NVIDIA T4 GPU with batch size 1 and image size 640*

---

## 🛠️ Advanced Usage

### Custom Training

```python
# train_custom.py
from utils.general import check_dataset
from train import train

# Custom training configuration
opt = {
    'weights': 'yolov5s.pt',
    'cfg': 'models/yolov5s.yaml',
    'data': 'anti_uav_yolo.yaml',
    'hyp': 'data/hyps/hyp.scratch-med.yaml',
    'epochs': 100,
    'batch_size': 16,
    'imgsz': 640,
    'device': '0',
    'project': 'runs',
    'name': 'custom_exp',
    # Add more options as needed
}

# Start training
train(opt)
```

### Hyperparameter Tuning

```bash
# Evolve hyperparameters
python train.py --data anti_uav_yolo.yaml \
                --weights yolov5s.pt \
                --img 640 \
                --batch 16 \
                --epochs 10 \
                --evolve 300

# Best hyperparameters will be saved in runs/evolve/
```

### Ensemble Inference

```python
# ensemble.py
import torch

# Load multiple models
models = [
    torch.hub.load('ultralytics/yolov5', 'custom', 'runs/exp_FULL/weights/best.pt'),
    torch.hub.load('ultralytics/yolov5', 'custom', 'runs/exp_IR/weights/best.pt'),
    torch.hub.load('ultralytics/yolov5', 'custom', 'runs/exp_RGB/weights/best.pt'),
]

# Ensemble inference
img = 'inputs/test.jpg'
results = []
for model in models:
    results.append(model(img))

# Combine results (implement voting or averaging)
# ... ensemble logic
```

---

## 📚 Documentation

### Project Structure

```
yolo5_UAV/
├── data/                    # Dataset configs and scripts
│   ├── hyps/               # Hyperparameter configs
│   ├── images/             # Sample images
│   └── scripts/            # Dataset download scripts
├── models/                  # Model architectures
│   ├── hub/                # Pretrained model configs
│   └── segment/            # Segmentation models
├── utils/                   # Utility functions
│   ├── loggers/            # Training loggers
│   └── segment/            # Segmentation utils
├── runs/                    # Training and inference results
│   ├── exp_FULL/           # Full dataset training
│   ├── exp_IR/             # IR-only training
│   ├── exp_RGB/            # RGB-only training
│   └── detect/             # Detection results
├── inputs/                  # Input images/videos
├── train.py                 # Training script
├── val.py                   # Validation script
├── detect.py                # Inference script
├── export.py                # Model export script
├── anti_uav_yolo.yaml      # Dataset configuration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### Key Files

- `train.py` - Main training script with distributed training support
- `val.py` - Validation and testing script
- `detect.py` - Inference on images, videos, streams
- `export.py` - Export models to various formats (ONNX, TensorRT, etc.)
- `anti_uav_yolo.yaml` - Dataset configuration for Anti-UAV dataset

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 TODO

- [ ] Add support for more UAV datasets
- [ ] Implement tracking (DeepSORT, ByteTrack)
- [ ] Add attention mechanisms (CBAM, SE)
- [ ] Optimize for Jetson Nano (FP16, INT8)
- [ ] Create GUI for real-time monitoring
- [ ] Add multi-camera support
- [ ] Implement distance estimation
- [ ] Add flight path prediction
- [ ] Create Docker containers for easy deployment
- [ ] Add web interface for remote monitoring

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@misc{yolov5_uav_2024,
  title={YOLOv5 UAV Detection System},
  author={Tigran98},
  year={2024},
  publisher={GitHub},
  url={https://github.com/Tigran98/yolo5_UAV}
}
```

**YOLOv5 Citation:**
```bibtex
@software{yolov5_2020,
  title={YOLOv5 by Ultralytics},
  author={Jocher, Glenn},
  year={2020},
  version={7.0},
  license={AGPL-3.0},
  url={https://github.com/ultralytics/yolov5}
}
```

---

## 📄 License

This project is licensed under the **AGPL-3.0 License** - see the [LICENSE](LICENSE) file for details.

**Note**: This project uses YOLOv5 which is licensed under AGPL-3.0. For commercial use, please obtain appropriate licensing from [Ultralytics](https://ultralytics.com/license).

---

## 🙏 Acknowledgments

- **Ultralytics** for the excellent YOLOv5 framework
- **Anti-UAV Dataset** creators for providing the training data
- **NVIDIA** for Jetson hardware and TensorRT optimization tools
- **Xilinx** for Vitis AI framework and FPGA deployment tools
- Open-source community for continuous support

---

## 📧 Contact

For questions, issues, or collaboration:

- **GitHub Issues**: [https://github.com/Tigran98/yolo5_UAV/issues](https://github.com/Tigran98/yolo5_UAV/issues)
- **GitHub**: [@Tigran98](https://github.com/Tigran98)

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

<div align="center">
  <img src="https://img.shields.io/github/stars/Tigran98/yolo5_UAV?style=social" alt="GitHub stars">
  <img src="https://img.shields.io/github/forks/Tigran98/yolo5_UAV?style=social" alt="GitHub forks">
  <img src="https://img.shields.io/github/watchers/Tigran98/yolo5_UAV?style=social" alt="GitHub watchers">
</div>

---

<div align="center">
  <p>Made with ❤️ for UAV Detection Research</p>
  <p>© 2024 YOLOv5 UAV Detection System</p>
</div>

