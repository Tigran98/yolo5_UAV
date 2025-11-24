import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def parse_yolo_label(label_line):
    """Parse YOLO format label"""
    parts = label_line.strip().split()
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

def align_rgb_to_ir_coordinate_space(rgb_img, ir_img, rgb_label, ir_label, target_size=640):
    """
    Align RGB image to IR coordinate space so a single label works for both.
    
    Strategy:
    1. Resize IR normally with letterbox
    2. Calculate where IR drone ends up after resize
    3. Adjust RGB transformation to place RGB drone at same location
    """
    rgb_h, rgb_w = rgb_img.shape[:2]
    ir_h, ir_w = ir_img.shape[:2]
    
    # Parse labels
    rgb_label_dict = parse_yolo_label(rgb_label)
    ir_label_dict = parse_yolo_label(ir_label)
    
    # Get drone positions in original images (pixel coords)
    rgb_drone_x, rgb_drone_y, rgb_drone_w, rgb_drone_h = yolo_to_pixels(rgb_label_dict, rgb_w, rgb_h)
    ir_drone_x, ir_drone_y, ir_drone_w, ir_drone_h = yolo_to_pixels(ir_label_dict, ir_w, ir_h)
    
    print(f"\nOriginal drone positions:")
    print(f"  RGB: ({rgb_drone_x:.1f}, {rgb_drone_y:.1f})")
    print(f"  IR:  ({ir_drone_x:.1f}, {ir_drone_y:.1f})")
    
    # Resize IR with standard letterbox
    ir_resized, ir_scale, (ir_pad_top, ir_pad_left) = letterbox_resize(ir_img, target_size)
    
    # Calculate where IR drone is after resize
    ir_drone_x_resized = ir_drone_x * ir_scale + ir_pad_left
    ir_drone_y_resized = ir_drone_y * ir_scale + ir_pad_top
    
    print(f"\nIR after letterbox:")
    print(f"  Scale: {ir_scale:.4f}")
    print(f"  Padding: ({ir_pad_top}px top, {ir_pad_left}px left)")
    print(f"  Drone position: ({ir_drone_x_resized:.1f}, {ir_drone_y_resized:.1f})")
    
    # Now we need RGB drone to end up at the same position
    # Working backwards: (rgb_drone_x_resized, rgb_drone_y_resized) = (ir_drone_x_resized, ir_drone_y_resized)
    # rgb_drone_x * scale + pad_left = ir_drone_x_resized
    # rgb_drone_y * scale + pad_top = ir_drone_y_resized
    
    # We have flexibility in choosing scale and padding for RGB
    # Let's try to match the IR scale if possible, or choose optimal scale
    
    # Option 1: Try to use same scale as IR (might not fit in target_size)
    # Option 2: Calculate required scale and padding to align drones
    
    # Let's use Option 2: Calculate custom transformation
    # We want: rgb_drone_y * scale + pad_top = ir_drone_y_resized
    
    # Strategy: Use standard letterbox for RGB, then calculate offset correction
    rgb_resized_standard, rgb_scale, (rgb_pad_top, rgb_pad_left) = letterbox_resize(rgb_img, target_size)
    
    rgb_drone_x_standard = rgb_drone_x * rgb_scale + rgb_pad_left
    rgb_drone_y_standard = rgb_drone_y * rgb_scale + rgb_pad_top
    
    print(f"\nRGB after standard letterbox:")
    print(f"  Scale: {rgb_scale:.4f}")
    print(f"  Padding: ({rgb_pad_top}px top, {rgb_pad_left}px left)")
    print(f"  Drone position: ({rgb_drone_x_standard:.1f}, {rgb_drone_y_standard:.1f})")
    
    # Calculate the offset needed
    offset_x = ir_drone_x_resized - rgb_drone_x_standard
    offset_y = ir_drone_y_resized - rgb_drone_y_standard
    
    print(f"\nOffset needed to align: ({offset_x:.1f}px, {offset_y:.1f}px)")
    
    # Apply offset by adjusting padding
    # We'll shift the RGB image by translating it
    M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
    rgb_resized_aligned = cv2.warpAffine(rgb_resized_standard, M, (target_size, target_size),
                                         borderMode=cv2.BORDER_CONSTANT, 
                                         borderValue=(114, 114, 114))
    
    # Verify alignment
    rgb_drone_x_aligned = rgb_drone_x_standard + offset_x
    rgb_drone_y_aligned = rgb_drone_y_standard + offset_y
    
    print(f"\nRGB after alignment:")
    print(f"  Drone position: ({rgb_drone_x_aligned:.1f}, {rgb_drone_y_aligned:.1f})")
    print(f"  Match with IR: {abs(rgb_drone_x_aligned - ir_drone_x_resized) < 1 and abs(rgb_drone_y_aligned - ir_drone_y_resized) < 1}")
    
    # Create unified label (use IR label since that's the target)
    unified_label = ir_label.strip()
    
    # Adjust IR label for the resized coordinates
    ir_drone_w_resized = ir_drone_w * ir_scale
    ir_drone_h_resized = ir_drone_h * ir_scale
    
    unified_label_dict = pixels_to_yolo(ir_drone_x_resized, ir_drone_y_resized,
                                       ir_drone_w_resized, ir_drone_h_resized,
                                       target_size, target_size)
    
    unified_label_str = f"0 {unified_label_dict['x']:.6f} {unified_label_dict['y']:.6f} {unified_label_dict['w']:.6f} {unified_label_dict['h']:.6f}"
    
    return rgb_resized_aligned, ir_resized, unified_label_str

def draw_yolo_box(image, label, color=(0, 255, 0), thickness=2, text=""):
    """Draw bounding box from YOLO label"""
    h, w = image.shape[:2]
    label_dict = parse_yolo_label(label)
    
    x_center = label_dict['x'] * w
    y_center = label_dict['y'] * h
    box_w = label_dict['w'] * w
    box_h = label_dict['h'] * h
    
    x1 = int(x_center - box_w / 2)
    y1 = int(y_center - box_h / 2)
    x2 = int(x_center + box_w / 2)
    y2 = int(y_center + box_h / 2)
    
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    cv2.circle(image, (int(x_center), int(y_center)), 4, (0, 0, 255), -1)
    
    if text:
        cv2.putText(image, text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return image

# Main execution
if __name__ == "__main__":
    # Load images
    rgb_path = r"D:\erkaki\yolov8_nano_ssd\data\anti-uav\images\train\20190925_130434_1_4_visible_385.jpg"
    ir_path = r"D:\erkaki\yolov8_nano_ssd\data\anti-uav\images\train\20190925_130434_1_4_infrared_385.jpg"
    rgb_label_path = r"D:\erkaki\yolov8_nano_ssd\data\anti-uav\labels\train\20190925_130434_1_4_visible_385.txt"
    ir_label_path = r"D:\erkaki\yolov8_nano_ssd\data\anti-uav\labels\train\20190925_130434_1_4_infrared_385.txt"
    
    rgb_img = cv2.imread(rgb_path)
    ir_img = cv2.imread(ir_path)
    
    with open(rgb_label_path, 'r') as f:
        rgb_label = f.readline()
    with open(ir_label_path, 'r') as f:
        ir_label = f.readline()
    
    print("="*60)
    print("RGB-to-IR Coordinate Space Alignment")
    print("="*60)
    print(f"Original RGB: {rgb_img.shape[:2]}")
    print(f"Original IR: {ir_img.shape[:2]}")
    print(f"RGB label: {rgb_label.strip()}")
    print(f"IR label: {ir_label.strip()}")
    
    # Align RGB to IR coordinate space
    rgb_aligned, ir_resized, unified_label = align_rgb_to_ir_coordinate_space(
        rgb_img, ir_img, rgb_label, ir_label, target_size=640
    )
    
    print(f"\n{'='*60}")
    print(f"Unified label: {unified_label}")
    print(f"{'='*60}")
    
    # Visualize
    rgb_vis = rgb_aligned.copy()
    ir_vis = ir_resized.copy()
    
    rgb_vis = draw_yolo_box(rgb_vis, unified_label, color=(0, 255, 0), text="Unified Label")
    ir_vis = draw_yolo_box(ir_vis, unified_label, color=(0, 255, 0), text="Unified Label")
    
    # Create comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Load originals for comparison
    rgb_orig = cv2.imread(rgb_path)
    ir_orig = cv2.imread(ir_path)
    rgb_orig = draw_yolo_box(rgb_orig, rgb_label, color=(255, 0, 0), text="RGB Label")
    rgb_orig = draw_yolo_box(rgb_orig, ir_label, color=(0, 0, 255), text="IR Label")
    ir_orig = draw_yolo_box(ir_orig, ir_label, color=(0, 255, 0), text="IR Label")
    
    axes[0, 0].imshow(cv2.cvtColor(rgb_orig, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('Original RGB\n(Red=RGB label, Blue=IR label)', fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(cv2.cvtColor(ir_orig, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('Original IR\n(Green=IR label)', fontsize=10, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(cv2.cvtColor(rgb_vis, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title('Aligned RGB (640×640)\n(Green=Unified label)', fontsize=10, fontweight='bold')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(cv2.cvtColor(ir_vis, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('Resized IR (640×640)\n(Green=Unified label)', fontsize=10, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save results
    output_dir = Path(r"D:\erkaki\yolov8_nano_ssd\data\anti-uav\tmp\resized_images")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    cv2.imwrite(str(output_dir / "rgb_aligned_640x640.jpg"), rgb_aligned)
    cv2.imwrite(str(output_dir / "ir_resized_640x640.jpg"), ir_resized)
    
    with open(output_dir / "unified_label.txt", 'w') as f:
        f.write(unified_label)
    
    plt.savefig(output_dir / "alignment_comparison.png", dpi=150, bbox_inches='tight')
    
    print(f"\n✅ Saved to {output_dir}")
    print(f"   - rgb_aligned_640x640.jpg")
    print(f"   - ir_resized_640x640.jpg")
    print(f"   - unified_label.txt")
    print(f"   - alignment_comparison.png")
