"""
Batch processor for Anti-UAV RGBT dataset with RGB-to-IR alignment
This ensures a single unified label works for both RGB and IR images
"""

import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import re

def parse_yolo_label(label_line):
    """Parse YOLO format label"""
    parts = label_line.strip().split()
    if len(parts) != 5:
        return None
    return {
        'class': int(parts[0]),
        'x': float(parts[1]),
        'y': float(parts[2]),
        'w': float(parts[3]),
        'h': float(parts[4])
    }

def yolo_to_pixels(label, img_w, img_h):
    """Convert YOLO normalized coords to pixel coords"""
    x_center = label['x'] * img_w
    y_center = label['y'] * img_h
    w = label['w'] * img_w
    h = label['h'] * img_h
    return x_center, y_center, w, h

def pixels_to_yolo(x_center, y_center, w, h, img_w, img_h):
    """Convert pixel coords to YOLO normalized coords"""
    return {
        'x': x_center / img_w,
        'y': y_center / img_h,
        'w': w / img_w,
        'h': h / img_h
    }

def letterbox_resize(image, target_size=640, color=(114, 114, 114)):
    """Standard letterbox resize"""
    h, w = image.shape[:2]
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                cv2.BORDER_CONSTANT, value=color)
    return padded, scale, (top, left)

def align_rgb_to_ir_batch(rgb_img, ir_img, rgb_label_line, ir_label_line, target_size=640):
    """
    Align RGB image to IR coordinate space for unified labeling
    Returns: rgb_aligned, ir_resized, unified_label_string
    """
    rgb_h, rgb_w = rgb_img.shape[:2]
    ir_h, ir_w = ir_img.shape[:2]
    
    # Parse labels
    rgb_label = parse_yolo_label(rgb_label_line)
    ir_label = parse_yolo_label(ir_label_line)
    
    if rgb_label is None or ir_label is None:
        return None, None, None
    
    # Get drone positions in original images
    rgb_drone_x, rgb_drone_y, rgb_drone_w, rgb_drone_h = yolo_to_pixels(rgb_label, rgb_w, rgb_h)
    ir_drone_x, ir_drone_y, ir_drone_w, ir_drone_h = yolo_to_pixels(ir_label, ir_w, ir_h)
    
    # Resize IR with standard letterbox (this is our reference)
    ir_resized, ir_scale, (ir_pad_top, ir_pad_left) = letterbox_resize(ir_img, target_size)
    
    # Calculate where IR drone ends up
    ir_drone_x_resized = ir_drone_x * ir_scale + ir_pad_left
    ir_drone_y_resized = ir_drone_y * ir_scale + ir_pad_top
    ir_drone_w_resized = ir_drone_w * ir_scale
    ir_drone_h_resized = ir_drone_h * ir_scale
    
    # Resize RGB with standard letterbox first
    rgb_resized_standard, rgb_scale, (rgb_pad_top, rgb_pad_left) = letterbox_resize(rgb_img, target_size)
    
    # Calculate where RGB drone is after standard letterbox
    rgb_drone_x_standard = rgb_drone_x * rgb_scale + rgb_pad_left
    rgb_drone_y_standard = rgb_drone_y * rgb_scale + rgb_pad_top
    
    # Calculate offset to align RGB drone with IR drone
    offset_x = ir_drone_x_resized - rgb_drone_x_standard
    offset_y = ir_drone_y_resized - rgb_drone_y_standard
    
    # Apply translation to align RGB
    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    rgb_aligned = cv2.warpAffine(rgb_resized_standard, M, (target_size, target_size),
                                 borderMode=cv2.BORDER_CONSTANT, 
                                 borderValue=(114, 114, 114))
    
    # Create unified label based on IR position
    unified_label_dict = pixels_to_yolo(ir_drone_x_resized, ir_drone_y_resized,
                                       ir_drone_w_resized, ir_drone_h_resized,
                                       target_size, target_size)
    
    unified_label = f"{ir_label['class']} {unified_label_dict['x']:.6f} {unified_label_dict['y']:.6f} {unified_label_dict['w']:.6f} {unified_label_dict['h']:.6f}"
    
    return rgb_aligned, ir_resized, unified_label

def process_set(root_dir, set_name, target_size=640):
    """
    Process a single set (e.g., train, val, test) with RGB-to-IR alignment.
    RGB and IR images/labels are in the same folders, differentiated by filenames:
    - RGB files contain "visible" in the name
    - IR files contain "infrared" in the name
    
    Args:
        root_dir: Root directory containing images/ and labels/ folders
        set_name: Name of the set folder (e.g., 'train', 'val', 'test')
        target_size: Target image size for resizing
    
    Returns:
        Statistics dictionary
    """
    root_dir = Path(root_dir)
    images_dir = root_dir / 'images'
    labels_dir = root_dir / 'labels'
    
    # Find set directories
    set_images_dir = images_dir / set_name
    set_labels_dir = labels_dir / set_name
    
    if not set_images_dir.exists() or not set_labels_dir.exists():
        return None
    
    # Create output directories
    output_set_name = f"{set_name}_resized"
    output_images_dir = images_dir / output_set_name
    output_labels_dir = labels_dir / output_set_name
    
    output_rgb_img_dir = output_images_dir / 'rgb'
    output_ir_img_dir = output_images_dir / 'ir'
    
    for dir_path in [output_rgb_img_dir, output_ir_img_dir, output_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Find all RGB images (containing "visible" in filename)
    all_images = sorted(list(set_images_dir.glob('*.jpg')) + 
                       list(set_images_dir.glob('*.png')) +
                       list(set_images_dir.glob('*.jpeg')))
    
    rgb_images = [img for img in all_images if 'visible' in img.name.lower()]
    
    print(f"Found {len(rgb_images)} RGB images (visible) in {set_name}")
    
    stats = {
        'processed': 0,
        'failed': 0,
        'missing_pairs': 0,
        'failed_reasons': {
            'image_load_error': 0,
            'empty_labels': 0,
            'invalid_label_format': 0,
            'alignment_error': 0,
            'other_exceptions': 0
        },
        'alignment_offsets': []
    }
    
    for rgb_img_path in tqdm(rgb_images, desc=f"Processing {set_name}"):
        try:
            # Extract base name and find corresponding IR image
            # Replace "visible" with "infrared" in filename
            rgb_filename = rgb_img_path.name
            ir_filename = rgb_filename.replace('visible', 'infrared')
            ir_img_path = set_images_dir / ir_filename
            
            # Get base name without extension for labels
            base_name = rgb_img_path.stem
            ir_base_name = base_name.replace('visible', 'infrared')
            
            # Find label files (same directory, same naming pattern)
            rgb_label_filename = f"{base_name}.txt"
            ir_label_filename = f"{ir_base_name}.txt"
            
            rgb_label_path = set_labels_dir / rgb_label_filename
            ir_label_path = set_labels_dir / ir_label_filename
            
            # Check if all files exist
            if not ir_img_path.exists():
                stats['missing_pairs'] += 1
                continue
            if not rgb_label_path.exists() or not ir_label_path.exists():
                stats['missing_pairs'] += 1
                continue
            
            # Load images
            rgb_img = cv2.imread(str(rgb_img_path))
            ir_img = cv2.imread(str(ir_img_path))
            
            if rgb_img is None or ir_img is None:
                stats['failed'] += 1
                stats['failed_reasons']['image_load_error'] += 1
                if rgb_img is None:
                    print(f"\nWarning: Could not load RGB image: {rgb_img_path.name}")
                if ir_img is None:
                    print(f"\nWarning: Could not load IR image: {ir_img_path.name}")
                continue
            
            # Load labels (only first line for now)
            with open(rgb_label_path, 'r') as f:
                rgb_labels = f.readlines()
            with open(ir_label_path, 'r') as f:
                ir_labels = f.readlines()
            
            if not rgb_labels or not ir_labels:
                stats['failed'] += 1
                stats['failed_reasons']['empty_labels'] += 1
                if not rgb_labels:
                    print(f"\nWarning: Empty RGB label file: {rgb_label_path.name}")
                if not ir_labels:
                    print(f"\nWarning: Empty IR label file: {ir_label_path.name}")
                continue
            
            # Process alignment (using first label)
            rgb_aligned, ir_resized, unified_label = align_rgb_to_ir_batch(
                rgb_img, ir_img, rgb_labels[0], ir_labels[0], target_size
            )
            
            if rgb_aligned is None:
                stats['failed'] += 1
                # Check which part of alignment failed
                rgb_label = parse_yolo_label(rgb_labels[0])
                ir_label = parse_yolo_label(ir_labels[0])
                if rgb_label is None or ir_label is None:
                    stats['failed_reasons']['invalid_label_format'] += 1
                    if rgb_label is None:
                        print(f"\nWarning: Invalid RGB label format: {rgb_label_path.name} - {rgb_labels[0].strip()}")
                    if ir_label is None:
                        print(f"\nWarning: Invalid IR label format: {ir_label_path.name} - {ir_labels[0].strip()}")
                else:
                    stats['failed_reasons']['alignment_error'] += 1
                    print(f"\nWarning: Alignment failed for {rgb_img_path.name}")
                continue
            
            # Save images with consistent naming (remove visible/infrared from output names)
            # Use a common base name for output - remove visible/infrared identifiers
            output_base_name = base_name
            # Remove visible/infrared (case-insensitive)
            output_base_name = re.sub(r'_?visible_?', '', output_base_name, flags=re.IGNORECASE)
            output_base_name = re.sub(r'_?infrared_?', '', output_base_name, flags=re.IGNORECASE)
            # Clean up any double underscores or trailing/leading underscores
            output_base_name = re.sub(r'_+', '_', output_base_name)
            output_base_name = output_base_name.strip('_')
            
            cv2.imwrite(str(output_rgb_img_dir / f"{output_base_name}.jpg"), rgb_aligned)
            cv2.imwrite(str(output_ir_img_dir / f"{output_base_name}.jpg"), ir_resized)
            
            # Save unified label (same for both RGB and IR)
            with open(output_labels_dir / f"{output_base_name}.txt", 'w') as f:
                f.write(unified_label)
            
            stats['processed'] += 1
            
        except Exception as e:
            print(f"\nError processing {rgb_img_path.name}: {str(e)}")
            stats['failed'] += 1
            stats['failed_reasons']['other_exceptions'] += 1
    
    return stats

def main():
    parser = argparse.ArgumentParser(
        description='Batch process Anti-UAV dataset with RGB-to-IR alignment for unified labels'
    )
    parser.add_argument('--root', type=str, required=True,
                       help='Root directory containing images/ and labels/ folders')
    parser.add_argument('--target-size', type=int, default=640,
                       help='Target image size (default: 640)')
    
    args = parser.parse_args()
    
    root_dir = Path(args.root)
    
    if not root_dir.exists():
        print(f"Error: Root directory {root_dir} does not exist")
        return
    
    images_dir = root_dir / 'images'
    labels_dir = root_dir / 'labels'
    
    if not images_dir.exists():
        print(f"Error: {images_dir} does not exist")
        return
    if not labels_dir.exists():
        print(f"Error: {labels_dir} does not exist")
        return
    
    # Find all set folders in images directory
    set_folders = [d.name for d in images_dir.iterdir() 
                   if d.is_dir() and not d.name.endswith('_resized')]
    
    if not set_folders:
        print(f"Error: No set folders found in {images_dir}")
        return
    
    print("="*60)
    print("Anti-UAV RGBT Dataset - RGB-to-IR Alignment Processor")
    print("Unified Labeling: One label works for both RGB and IR")
    print("="*60)
    print(f"\nFound {len(set_folders)} set(s): {', '.join(set_folders)}")
    
    all_stats = {}
    
    # Process each set
    for i, set_name in enumerate(sorted(set_folders), 1):
        print(f"\n[{i}/{len(set_folders)}] Processing {set_name} set...")
        stats = process_set(root_dir, set_name, args.target_size)
        
        if stats is not None:
            all_stats[set_name] = stats
            print(f"\n{set_name} set statistics:")
            print(f"  - Processed: {stats['processed']}")
            print(f"  - Failed: {stats['failed']}")
            if stats['failed'] > 0:
                print(f"    Failure breakdown:")
                for reason, count in stats['failed_reasons'].items():
                    if count > 0:
                        reason_name = reason.replace('_', ' ').title()
                        print(f"      - {reason_name}: {count}")
            print(f"  - Missing pairs: {stats['missing_pairs']}")
        else:
            print(f"  ⚠ Skipped {set_name} (missing RGB/IR directories or labels)")
    
    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)
    print("\nOutput structure:")
    print(f"  {root_dir}/")
    print(f"    ├── images/")
    for set_name in sorted(set_folders):
        print(f"    │   ├── {set_name}_resized/")
        print(f"    │   │   ├── rgb/     (aligned RGB images)")
        print(f"    │   │   └── ir/      (resized IR images)")
    print(f"    └── labels/")
    for set_name in sorted(set_folders):
        print(f"        └── {set_name}_resized/  (unified labels)")
    
    # Save processing info
    info = {
        'method': 'RGB-to-IR alignment',
        'description': 'RGB images aligned to IR coordinate space for unified labeling',
        'target_size': args.target_size,
        'root_directory': str(root_dir),
        'sets_processed': list(all_stats.keys()),
        'statistics': all_stats
    }
    
    with open(root_dir / 'processing_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"\n✅ Processing info saved to: {root_dir / 'processing_info.json'}")

if __name__ == "__main__":
    main()
