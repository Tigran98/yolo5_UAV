"""
Comprehensive tests for fusion model data loader and loss computation.
Validates:
1. Label file loading and parsing
2. Image pair loading (RGB + IR)
3. Label format validation (normalized xywh)
4. Label coordinate transformations
5. Dataset __getitem__ correctness
6. Collate function correctness
7. Loss computation correctness
"""

import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
import yaml

# Add yolov5 to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.fusion_dataloaders import DualModalDataset, create_fusion_dataloader
from utils.general import xywhn2xyxy, xyxy2xywhn
from utils.loss import ComputeLoss
from models.fusion_model import FusionModel

# Test configuration
DATASET_PATH = r"D:\erkaki\yolov8_nano_ssd\data\anti-uav"
TRAIN_IMAGES = os.path.join(DATASET_PATH, "images", "test")
TRAIN_LABELS = os.path.join(DATASET_PATH, "labels", "test")


def test_label_file_format():
    """Test that label files are in correct YOLO format: class x y w h (normalized)"""
    print("\n=== Test 1: Label File Format ===")
    
    import glob
    label_files = glob.glob(os.path.join(TRAIN_LABELS, "*.txt"))[:10]
    
    all_valid = True
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.read().strip().splitlines()
            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"ERROR: {os.path.basename(label_file)} line {line_num}: Expected 5 values, got {len(parts)}")
                    all_valid = False
                    continue
                
                try:
                    cls = int(parts[0])
                    x, y, w, h = map(float, parts[1:])
                    
                    # Validate normalized coordinates
                    if not (0 <= x <= 1 and 0 <= y <= 1):
                        print(f"ERROR: {os.path.basename(label_file)} line {line_num}: x,y not normalized: ({x}, {y})")
                        all_valid = False
                    if not (0 < w <= 1 and 0 < h <= 1):
                        print(f"ERROR: {os.path.basename(label_file)} line {line_num}: w,h not normalized: ({w}, {h})")
                        all_valid = False
                    if cls != 0:
                        print(f"WARNING: {os.path.basename(label_file)} line {line_num}: Expected class 0, got {cls}")
                        
                except ValueError as e:
                    print(f"ERROR: {os.path.basename(label_file)} line {line_num}: Invalid format: {e}")
                    all_valid = False
    
    if all_valid:
        print("✓ All label files are in correct YOLO format")
    else:
        print("✗ Some label files have format issues")
    
    return all_valid


def test_image_pair_matching():
    """Test that RGB and IR image pairs are correctly matched"""
    print("\n=== Test 2: Image Pair Matching ===")
    
    import glob
    rgb_files = sorted(glob.glob(os.path.join(TRAIN_IMAGES, "*_visible_*.jpg")))[:10]
    
    all_matched = True
    for rgb_file in rgb_files:
        ir_file = rgb_file.replace("_visible_", "_infrared_")
        if not os.path.exists(ir_file):
            print(f"ERROR: No IR pair found for {os.path.basename(rgb_file)}")
            all_matched = False
            continue
        
        # Load images to verify they exist and are valid
        rgb_img = cv2.imread(rgb_file)
        ir_img = cv2.imread(ir_file)
        
        if rgb_img is None:
            print(f"ERROR: Cannot load RGB image: {os.path.basename(rgb_file)}")
            all_matched = False
        if ir_img is None:
            print(f"ERROR: Cannot load IR image: {os.path.basename(ir_file)}")
            all_matched = False
        
        if rgb_img is not None and ir_img is not None:
            if rgb_img.shape[:2] != ir_img.shape[:2]:
                print(f"WARNING: {os.path.basename(rgb_file)}: RGB and IR shapes differ: {rgb_img.shape[:2]} vs {ir_img.shape[:2]}")
    
    if all_matched:
        print(f"✓ All {len(rgb_files)} RGB-IR pairs are correctly matched")
    else:
        print("✗ Some image pairs are missing or invalid")
    
    return all_matched


def test_label_coordinate_transformations():
    """Test label coordinate transformations: xywhn <-> xyxy"""
    print("\n=== Test 3: Label Coordinate Transformations ===")
    
    # Test with known values
    img_w, img_h = 640, 480
    labels_xywhn = np.array([
        [0, 0.5, 0.5, 0.2, 0.2],  # Center, 20% width/height
        [0, 0.1, 0.1, 0.1, 0.1],  # Top-left corner
        [0, 0.9, 0.9, 0.1, 0.1],  # Bottom-right corner
    ], dtype=np.float32)
    
    # Convert to xyxy
    labels_xyxy = xywhn2xyxy(labels_xywhn[:, 1:], img_w, img_h)
    
    # Convert back to xywhn
    labels_xywhn_recovered = xyxy2xywhn(labels_xyxy, img_w, img_h)
    
    # Check if recovered values match original (within tolerance)
    tolerance = 1e-5
    diff = np.abs(labels_xywhn[:, 1:] - labels_xywhn_recovered)
    
    if np.all(diff < tolerance):
        print("✓ Coordinate transformations are correct (round-trip test)")
        print(f"  Max difference: {diff.max():.2e}")
        return True
    else:
        print("✗ Coordinate transformations have errors")
        print(f"  Max difference: {diff.max():.2e}")
        print(f"  Original: {labels_xywhn[0, 1:]}")
        print(f"  Recovered: {labels_xywhn_recovered[0]}")
        return False


def test_dataset_initialization():
    """Test DualModalDataset initialization"""
    print("\n=== Test 4: Dataset Initialization ===")
    
    try:
        from utils.fusion_dataloaders import DualModalDataset
        import yaml
        
        # Load dataset config
        with open("data/anti_uav_fusion.yaml", 'r') as f:
            data_cfg = yaml.safe_load(f)
        
        dataset_path = Path(data_cfg['path'])
        train_path = dataset_path / data_cfg['train']
        
        # Create dataset
        dataset = DualModalDataset(
            path=str(train_path),
            img_size=256,
            batch_size=2,
            augment=False,  # Disable augmentation for testing
            hyp={
                "mosaic": 0.0,
                "mixup": 0.0,
                "degrees": 0.0,
                "translate": 0.0,
                "scale": 0.0,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.0,
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
            },
            rect=False,
            cache_images=False,
            single_cls=False,
            stride=32,
            prefix="test: ",
        )
        
        print(f"✓ Dataset initialized successfully")
        print(f"  Total samples: {len(dataset)}")
        print(f"  RGB files: {len(dataset.rgb_files)}")
        print(f"  IR files: {len(dataset.ir_files)}")
        print(f"  Labels loaded: {len(dataset.labels)}")
        
        # Check that all arrays have same length
        if len(dataset.rgb_files) == len(dataset.ir_files) == len(dataset.labels):
            print("✓ All dataset arrays have consistent length")
            return True
        else:
            print(f"✗ Array length mismatch: RGB={len(dataset.rgb_files)}, IR={len(dataset.ir_files)}, Labels={len(dataset.labels)}")
            return False
            
    except Exception as e:
        print(f"✗ Dataset initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_getitem():
    """Test dataset __getitem__ returns correct format"""
    print("\n=== Test 5: Dataset __getitem__ ===")
    
    try:
        from utils.fusion_dataloaders import DualModalDataset
        import yaml
        
        # Load dataset config
        with open("data/anti_uav_fusion.yaml", 'r') as f:
            data_cfg = yaml.safe_load(f)
        
        dataset_path = Path(data_cfg['path'])
        train_path = dataset_path / data_cfg['train']
        
        # Create dataset
        dataset = DualModalDataset(
            path=str(train_path),
            img_size=256,
            batch_size=2,
            augment=False,  # Disable augmentation for testing
            hyp={
                "mosaic": 0.0,
                "mixup": 0.0,
                "degrees": 0.0,
                "translate": 0.0,
                "scale": 0.0,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.0,
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
            },
            rect=False,
            cache_images=False,
            single_cls=False,
            stride=32,
            prefix="test: ",
        )
        
        # Get a sample
        rgb_img, ir_img, labels, paths, shapes = dataset[0]
        
        # Validate outputs
        checks = []
        
        # Check RGB image
        if isinstance(rgb_img, torch.Tensor) and rgb_img.shape == (3, 256, 256):
            checks.append("✓ RGB image: correct shape (3, 256, 256)")
        else:
            checks.append(f"✗ RGB image: wrong shape {rgb_img.shape if isinstance(rgb_img, torch.Tensor) else type(rgb_img)}")
        
        # Check IR image
        if isinstance(ir_img, torch.Tensor) and ir_img.shape == (3, 256, 256):
            checks.append("✓ IR image: correct shape (3, 256, 256)")
        else:
            checks.append(f"✗ IR image: wrong shape {ir_img.shape if isinstance(ir_img, torch.Tensor) else type(ir_img)}")
        
        # Check labels
        if isinstance(labels, torch.Tensor):
            if labels.shape[1] == 6:  # [batch_idx, cls, x, y, w, h]
                checks.append(f"✓ Labels: correct shape (N, 6), got {labels.shape}")
                
                # Check normalized coordinates
                if labels.shape[0] > 0:
                    coords = labels[:, 2:6]  # x, y, w, h
                    if torch.all(coords >= 0) and torch.all(coords <= 1):
                        checks.append("✓ Labels: coordinates are normalized (0-1)")
                    else:
                        checks.append(f"✗ Labels: coordinates not normalized: min={coords.min():.3f}, max={coords.max():.3f}")
            else:
                checks.append(f"✗ Labels: wrong shape {labels.shape}, expected (N, 6)")
        else:
            checks.append(f"✗ Labels: not a tensor, got {type(labels)}")
        
        # Check paths
        if isinstance(paths, tuple) and len(paths) == 2:
            checks.append("✓ Paths: correct format (rgb_path, ir_path)")
        else:
            checks.append(f"✗ Paths: wrong format {type(paths)}")
        
        # Check shapes
        if shapes is not None:
            checks.append("✓ Shapes: provided")
        else:
            checks.append("✗ Shapes: None")
        
        for check in checks:
            print(f"  {check}")
        
        all_passed = all("✓" in check for check in checks)
        return all_passed
        
    except Exception as e:
        print(f"✗ Dataset __getitem__ failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_collate_fn():
    """Test collate function correctness"""
    print("\n=== Test 6: Collate Function ===")
    
    try:
        from utils.fusion_dataloaders import DualModalDataset
        import yaml
        
        # Load dataset config
        with open("data/anti_uav_fusion.yaml", 'r') as f:
            data_cfg = yaml.safe_load(f)
        
        dataset_path = Path(data_cfg['path'])
        train_path = dataset_path / data_cfg['train']
        
        # Create dataset
        dataset = DualModalDataset(
            path=str(train_path),
            img_size=256,
            batch_size=2,
            augment=False,
            hyp={
                "mosaic": 0.0,
                "mixup": 0.0,
                "degrees": 0.0,
                "translate": 0.0,
                "scale": 0.0,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.0,
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
            },
            rect=False,
            cache_images=False,
            single_cls=False,
            stride=32,
            prefix="test: ",
        )
        
        # Get a batch
        batch = [dataset[i] for i in range(min(2, len(dataset)))]
        
        # Collate
        rgb_imgs, ir_imgs, labels, paths, shapes = dataset.collate_fn(batch)
        
        checks = []
        
        # Check RGB images batch
        if rgb_imgs.shape[0] == len(batch) and rgb_imgs.shape[1:] == (3, 256, 256):
            checks.append(f"✓ RGB batch: correct shape {rgb_imgs.shape}")
        else:
            checks.append(f"✗ RGB batch: wrong shape {rgb_imgs.shape}")
        
        # Check IR images batch
        if ir_imgs.shape[0] == len(batch) and ir_imgs.shape[1:] == (3, 256, 256):
            checks.append(f"✓ IR batch: correct shape {ir_imgs.shape}")
        else:
            checks.append(f"✗ IR batch: wrong shape {ir_imgs.shape}")
        
        # Check labels
        if isinstance(labels, torch.Tensor):
            if labels.shape[1] == 6:
                # Check that batch indices are correct
                unique_batch_indices = labels[:, 0].unique().tolist()
                expected_indices = list(range(len(batch)))
                if unique_batch_indices == expected_indices:
                    checks.append(f"✓ Labels: batch indices correct {unique_batch_indices}")
                else:
                    checks.append(f"✗ Labels: batch indices wrong {unique_batch_indices}, expected {expected_indices}")
                
                # Check normalized coordinates
                if labels.shape[0] > 0:
                    coords = labels[:, 2:6]
                    if torch.all(coords >= 0) and torch.all(coords <= 1):
                        checks.append("✓ Labels: coordinates normalized")
                    else:
                        checks.append(f"✗ Labels: coordinates not normalized")
            else:
                checks.append(f"✗ Labels: wrong shape {labels.shape}")
        else:
            checks.append(f"✗ Labels: not a tensor")
        
        for check in checks:
            print(f"  {check}")
        
        all_passed = all("✓" in check for check in checks)
        return all_passed
        
    except Exception as e:
        print(f"✗ Collate function test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation():
    """Test loss computation with known inputs"""
    print("\n=== Test 7: Loss Computation ===")
    
    try:
        import torch
        import yaml
        from models.fusion_model import FusionModel
        from utils.loss import ComputeLoss
        
        # Load hyperparameters
        with open("data/hyps/hyp.scratch-low.yaml", 'r') as f:
            hyp = yaml.safe_load(f)
        
        # Create a minimal model
        device = torch.device('cpu')
        model = FusionModel('models/yolov5n_fusion.yaml', ch=3, nc=1).to(device)
        model.hyp = hyp  # Set hyperparameters for loss computation
        model.train()  # Set to training mode for loss computation
        
        # Create loss computer
        compute_loss = ComputeLoss(model)
        
        # Create dummy inputs
        batch_size = 2
        rgb_imgs = torch.randn(batch_size, 3, 256, 256).to(device)
        ir_imgs = torch.randn(batch_size, 3, 256, 256).to(device)
        
        # Forward pass - need to normalize inputs (0-255 -> 0-1)
        rgb_imgs = rgb_imgs.float() / 255.0
        ir_imgs = ir_imgs.float() / 255.0
        
        # Create dummy targets: [batch_idx, cls, x, y, w, h] (normalized)
        targets = torch.tensor([
            [0, 0, 0.5, 0.5, 0.2, 0.2],  # Image 0, center box
            [1, 0, 0.3, 0.3, 0.1, 0.1],  # Image 1, small box
        ], dtype=torch.float32).to(device)
        
        # Forward pass (in training mode, returns list of tensors)
        predictions = model(rgb_imgs, ir_imgs)
        
        # Debug: Check prediction shapes
        if len(predictions) > 0:
            print(f"  Prediction shapes: {[p.shape if isinstance(p, torch.Tensor) else type(p) for p in predictions]}")
        
        # Ensure predictions is a list of tensors (training mode)
        if not isinstance(predictions, list):
            predictions = [predictions] if isinstance(predictions, torch.Tensor) else list(predictions)
        
        # Compute loss
        loss, loss_items = compute_loss(predictions, targets)
        
        checks = []
        
        # Check loss is a scalar tensor
        if isinstance(loss, torch.Tensor) and loss.numel() == 1:
            checks.append(f"✓ Total loss: scalar tensor, value={loss.item():.4f}")
        else:
            checks.append(f"✗ Total loss: wrong format {type(loss)}")
        
        # Check loss components
        if isinstance(loss_items, torch.Tensor) and loss_items.shape[0] >= 3:
            box_loss = loss_items[0]
            obj_loss = loss_items[1]
            cls_loss = loss_items[2]
            
            checks.append(f"✓ Loss items: box={box_loss.item():.4f}, obj={obj_loss.item():.4f}, cls={cls_loss.item():.4f}")
            
            # Check losses are non-negative
            if box_loss >= 0 and obj_loss >= 0 and cls_loss >= 0:
                checks.append("✓ All loss components are non-negative")
            else:
                checks.append("✗ Some loss components are negative")
            
            # Check total loss is sum of components multiplied by batch size
            # ComputeLoss returns (lbox + lobj + lcls) * bs, where bs is batch size
            expected_total = (box_loss + obj_loss + cls_loss) * batch_size
            if torch.abs(loss - expected_total) < 1e-4:
                checks.append("✓ Total loss matches sum of components (multiplied by batch size)")
            else:
                checks.append(f"✗ Total loss mismatch: {loss.item():.4f} vs {expected_total.item():.4f}")
                checks.append(f"  Note: Loss is multiplied by batch size ({batch_size})")
        else:
            checks.append(f"✗ Loss items: wrong format {type(loss_items)}")
        
        for check in checks:
            print(f"  {check}")
        
        all_passed = all("✓" in check for check in checks)
        return all_passed
        
    except Exception as e:
        print(f"✗ Loss computation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_label_loading_matches_original():
    """Test that labels loaded by dataset match original label files"""
    print("\n=== Test 8: Label Loading Accuracy ===")
    
    try:
        import glob
        from utils.fusion_dataloaders import DualModalDataset
        import yaml
        
        # Load dataset config
        with open("data/anti_uav_fusion.yaml", 'r') as f:
            data_cfg = yaml.safe_load(f)
        
        dataset_path = Path(data_cfg['path'])
        train_path = dataset_path / data_cfg['train']
        
        # Create dataset
        dataset = DualModalDataset(
            path=str(train_path),
            img_size=256,
            batch_size=2,
            augment=False,
            hyp={
                "mosaic": 0.0,
                "mixup": 0.0,
                "degrees": 0.0,
                "translate": 0.0,
                "scale": 0.0,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.0,
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
            },
            rect=False,
            cache_images=False,
            single_cls=False,
            stride=32,
            prefix="test: ",
        )
        
        # Test first few samples
        num_test = min(10, len(dataset))
        all_match = True
        
        for idx in range(num_test):
            rgb_path = dataset.rgb_files[idx]
            rgb_stem = Path(rgb_path).stem
            
            # Get expected label file path (prefer _infrared)
            label_file_base = rgb_stem.replace("_visible", "_infrared")
            label_file = os.path.join(TRAIN_LABELS, f"{label_file_base}.txt")
            
            if not os.path.exists(label_file):
                # Try without replacement
                label_file = os.path.join(TRAIN_LABELS, f"{rgb_stem}.txt")
            
            if os.path.exists(label_file):
                # Load original labels
                with open(label_file, 'r') as f:
                    original_labels = []
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            original_labels.append([float(x) for x in parts])
                    original_labels = np.array(original_labels, dtype=np.float32)
                
                # Get dataset labels
                dataset_labels = dataset.labels[idx]
                
                if len(original_labels) == len(dataset_labels):
                    # Compare (within tolerance due to potential rounding)
                    diff = np.abs(original_labels - dataset_labels)
                    max_diff = diff.max()
                    if max_diff < 1e-4:
                        continue  # Matches
                    else:
                        print(f"  ✗ Sample {idx}: Label mismatch (max diff: {max_diff:.2e})")
                        all_match = False
                else:
                    print(f"  ✗ Sample {idx}: Label count mismatch: {len(original_labels)} vs {len(dataset_labels)}")
                    all_match = False
        
        if all_match:
            print(f"✓ All {num_test} samples: Labels match original files")
        else:
            print(f"✗ Some samples have label mismatches")
        
        return all_match
        
    except Exception as e:
        print(f"✗ Label loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Fusion Model Data Loader & Loss Validation Tests")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Label File Format", test_label_file_format()))
    results.append(("Image Pair Matching", test_image_pair_matching()))
    results.append(("Coordinate Transformations", test_label_coordinate_transformations()))
    results.append(("Dataset Initialization", test_dataset_initialization()))
    results.append(("Dataset __getitem__", test_dataset_getitem()))
    results.append(("Collate Function", test_collate_fn()))
    results.append(("Loss Computation", test_loss_computation()))
    results.append(("Label Loading Accuracy", test_label_loading_matches_original()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! Data loader and loss computation are correct.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    exit(main())

