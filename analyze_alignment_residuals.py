"""
Analyze alignment residuals between RGB and IR labels.

For each paired RGB/IR image, computes the offset between the object's
position in RGB vs IR (in normalized coordinates). If the offset is
near-constant across all samples, it proves the displacement is a fixed
sensor calibration property, not scene-dependent.
"""

import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# ── Configuration ──────────────────────────────────────────────
IMG_DIR = r"D:\erkaki\yolov8_nano_ssd\data\anti-uav\images\val"
LBL_DIR = r"D:\erkaki\yolov8_nano_ssd\data\anti-uav\labels\val"
OUTPUT_DIR = r"D:\erkaki\yolov5_backup\alignment_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Image resolutions (from inspection above)
RGB_W, RGB_H = 1920, 1080
IR_W, IR_H = 640, 512


def parse_label(filepath):
    """Parse YOLO label file. Returns list of (cls, xc, yc, w, h) in normalized coords."""
    labels = []
    if not os.path.exists(filepath):
        return labels
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                labels.append((cls, xc, yc, w, h))
    return labels


def extract_base_and_frame(filename):
    """
    Extract base sequence name and frame number from filename.
    e.g., '20190925_134301_1_2_visible_0' -> ('20190925_134301_1_2', 0)
    """
    stem = Path(filename).stem
    if '_visible_' in stem:
        parts = stem.split('_visible_')
        return parts[0], int(parts[1])
    elif '_infrared_' in stem:
        parts = stem.split('_infrared_')
        return parts[0], int(parts[1])
    return None, None


# ── Discover and match RGB-IR label pairs ──────────────────────
print("Discovering label pairs...")
all_labels = glob.glob(os.path.join(LBL_DIR, "*.txt"))

rgb_labels = {}  # (base, frame) -> filepath
ir_labels = {}   # (base, frame) -> filepath

for lbl_path in all_labels:
    fname = os.path.basename(lbl_path)
    base, frame = extract_base_and_frame(fname)
    if base is None:
        continue
    if '_visible_' in fname:
        rgb_labels[(base, frame)] = lbl_path
    elif '_infrared_' in fname:
        ir_labels[(base, frame)] = lbl_path

# Match pairs
matched_keys = set(rgb_labels.keys()) & set(ir_labels.keys())
print(f"Found {len(rgb_labels)} RGB labels, {len(ir_labels)} IR labels")
print(f"Matched pairs: {len(matched_keys)}")

# ── Compute offsets ────────────────────────────────────────────
print("\nComputing per-sample offsets...")

# Offsets in normalized coordinates
dx_norm_list = []  # x offset (normalized)
dy_norm_list = []  # y offset (normalized)

# Offsets in pixel coordinates (after converting both to a common space)
dx_px_list = []  # x offset in pixels (relative to IR resolution)
dy_px_list = []  # y offset in pixels (relative to IR resolution)

# Per-sequence tracking
sequence_offsets = defaultdict(lambda: {'dx': [], 'dy': []})

skipped_empty = 0
skipped_multi = 0
processed = 0

for key in sorted(matched_keys):
    base, frame = key
    rgb_lbl = parse_label(rgb_labels[key])
    ir_lbl = parse_label(ir_labels[key])

    # Skip if either has no labels or different number of objects
    if len(rgb_lbl) == 0 or len(ir_lbl) == 0:
        skipped_empty += 1
        continue

    # Use first object (single-class drone detection — typically 1 object per frame)
    if len(rgb_lbl) != 1 or len(ir_lbl) != 1:
        skipped_multi += 1
        # Still process, use first object
    
    _, rgb_xc, rgb_yc, rgb_w, rgb_h = rgb_lbl[0]
    _, ir_xc, ir_yc, ir_w, ir_h = ir_lbl[0]

    # Offset in normalized coordinates
    dx_norm = ir_xc - rgb_xc
    dy_norm = ir_yc - rgb_yc
    dx_norm_list.append(dx_norm)
    dy_norm_list.append(dy_norm)

    # Convert to pixel coordinates in their respective images, 
    # then convert RGB pixel coords to IR-equivalent space
    rgb_xc_px = rgb_xc * RGB_W
    rgb_yc_px = rgb_yc * RGB_H
    ir_xc_px = ir_xc * IR_W
    ir_yc_px = ir_yc * IR_H

    # Scale RGB pixel coords to IR image space for comparison
    # (how many IR pixels apart are they in equivalent space)
    rgb_xc_in_ir_space = rgb_xc_px * (IR_W / RGB_W)
    rgb_yc_in_ir_space = rgb_yc_px * (IR_H / RGB_H)
    
    dx_px = ir_xc_px - rgb_xc_in_ir_space
    dy_px = ir_yc_px - rgb_yc_in_ir_space
    dx_px_list.append(dx_px)
    dy_px_list.append(dy_px)

    sequence_offsets[base]['dx'].append(dx_px)
    sequence_offsets[base]['dy'].append(dy_px)
    
    processed += 1

print(f"\nProcessed: {processed}")
print(f"Skipped (empty labels): {skipped_empty}")
print(f"Skipped (multi-object, still processed first): {skipped_multi}")

dx_norm_arr = np.array(dx_norm_list)
dy_norm_arr = np.array(dy_norm_list)
dx_px_arr = np.array(dx_px_list)
dy_px_arr = np.array(dy_px_list)

# ── Statistics ─────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ALIGNMENT RESIDUAL STATISTICS")
print("=" * 70)

print("\n--- Normalized Coordinate Offsets (IR - RGB) ---")
print(f"  dx: mean={dx_norm_arr.mean():.5f}, std={dx_norm_arr.std():.5f}, "
      f"min={dx_norm_arr.min():.5f}, max={dx_norm_arr.max():.5f}")
print(f"  dy: mean={dy_norm_arr.mean():.5f}, std={dy_norm_arr.std():.5f}, "
      f"min={dy_norm_arr.min():.5f}, max={dy_norm_arr.max():.5f}")

print("\n--- Pixel Offsets in IR Space (IR_px - RGB_scaled_to_IR_px) ---")
print(f"  dx: mean={dx_px_arr.mean():.2f} px, std={dx_px_arr.std():.2f} px, "
      f"min={dx_px_arr.min():.2f} px, max={dx_px_arr.max():.2f} px")
print(f"  dy: mean={dy_px_arr.mean():.2f} px, std={dy_px_arr.std():.2f} px, "
      f"min={dy_px_arr.min():.2f} px, max={dy_px_arr.max():.2f} px")

# What percentage fall within ±N pixels of the mean
for threshold in [2, 5, 10, 20]:
    within_x = np.abs(dx_px_arr - dx_px_arr.mean()) < threshold
    within_y = np.abs(dy_px_arr - dy_px_arr.mean()) < threshold
    within_both = within_x & within_y
    pct = 100 * within_both.sum() / len(within_both)
    print(f"  Within ±{threshold}px of mean offset: {pct:.1f}%")

# ── Plots ──────────────────────────────────────────────────────
print("\nGenerating plots...")

# 1. Histogram of dx and dy offsets (pixel space)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(dx_px_arr, bins=80, color='steelblue', edgecolor='black', alpha=0.8)
axes[0].axvline(dx_px_arr.mean(), color='red', linewidth=2, linestyle='--',
                label=f'Mean = {dx_px_arr.mean():.2f} px')
axes[0].axvline(dx_px_arr.mean() - dx_px_arr.std(), color='orange', linewidth=1.5, linestyle=':',
                label=f'Std = {dx_px_arr.std():.2f} px')
axes[0].axvline(dx_px_arr.mean() + dx_px_arr.std(), color='orange', linewidth=1.5, linestyle=':')
axes[0].set_xlabel('X Offset (pixels in IR space)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Horizontal Alignment Offset (IR - RGB)', fontsize=13)
axes[0].legend(fontsize=11)

axes[1].hist(dy_px_arr, bins=80, color='coral', edgecolor='black', alpha=0.8)
axes[1].axvline(dy_px_arr.mean(), color='red', linewidth=2, linestyle='--',
                label=f'Mean = {dy_px_arr.mean():.2f} px')
axes[1].axvline(dy_px_arr.mean() - dy_px_arr.std(), color='orange', linewidth=1.5, linestyle=':',
                label=f'Std = {dy_px_arr.std():.2f} px')
axes[1].axvline(dy_px_arr.mean() + dy_px_arr.std(), color='orange', linewidth=1.5, linestyle=':')
axes[1].set_xlabel('Y Offset (pixels in IR space)', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Vertical Alignment Offset (IR - RGB)', fontsize=13)
axes[1].legend(fontsize=11)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'offset_histograms.png'), dpi=150)
plt.close()
print(f"  Saved: offset_histograms.png")

# 2. 2D scatter plot of (dx, dy) offsets
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(dx_px_arr, dy_px_arr, alpha=0.15, s=8, c='steelblue')
ax.scatter([dx_px_arr.mean()], [dy_px_arr.mean()], color='red', s=200, marker='+',
           linewidths=3, zorder=5, label=f'Mean ({dx_px_arr.mean():.1f}, {dy_px_arr.mean():.1f})')

# Draw std ellipse
from matplotlib.patches import Ellipse
for n_std in [1, 2, 3]:
    ellipse = Ellipse(xy=(dx_px_arr.mean(), dy_px_arr.mean()),
                      width=2 * n_std * dx_px_arr.std(),
                      height=2 * n_std * dy_px_arr.std(),
                      fill=False, edgecolor='red', linestyle='--', linewidth=1.5,
                      label=f'{n_std}σ' if n_std == 1 else f'{n_std}σ')
    ax.add_patch(ellipse)

ax.set_xlabel('X Offset (pixels in IR space)', fontsize=12)
ax.set_ylabel('Y Offset (pixels in IR space)', fontsize=12)
ax.set_title('2D Alignment Offset Distribution (IR - RGB)', fontsize=13)
ax.legend(fontsize=11)
ax.set_aspect('equal')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'offset_scatter_2d.png'), dpi=150)
plt.close()
print(f"  Saved: offset_scatter_2d.png")

# 3. Per-sequence box plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
seq_names = sorted(sequence_offsets.keys())
dx_per_seq = [sequence_offsets[s]['dx'] for s in seq_names]
dy_per_seq = [sequence_offsets[s]['dy'] for s in seq_names]
short_names = [s[-10:] for s in seq_names]  # Shortened for labels

axes[0].boxplot(dx_per_seq, labels=short_names, vert=True)
axes[0].set_ylabel('X Offset (px)', fontsize=11)
axes[0].set_title('X Offset Distribution per Sequence', fontsize=12)
axes[0].tick_params(axis='x', rotation=45)
axes[0].axhline(dx_px_arr.mean(), color='red', linestyle='--', alpha=0.5)

axes[1].boxplot(dy_per_seq, labels=short_names, vert=True)
axes[1].set_ylabel('Y Offset (px)', fontsize=11)
axes[1].set_title('Y Offset Distribution per Sequence', fontsize=12)
axes[1].tick_params(axis='x', rotation=45)
axes[1].axhline(dy_px_arr.mean(), color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'offset_per_sequence.png'), dpi=150)
plt.close()
print(f"  Saved: offset_per_sequence.png")

# 4. Offset over time (frame index) for a single sequence
if seq_names:
    longest_seq = max(seq_names, key=lambda s: len(sequence_offsets[s]['dx']))
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    frames_dx = sequence_offsets[longest_seq]['dx']
    frames_dy = sequence_offsets[longest_seq]['dy']
    
    axes[0].plot(frames_dx, 'b-', alpha=0.6, linewidth=0.8)
    axes[0].axhline(np.mean(frames_dx), color='red', linestyle='--', linewidth=2,
                    label=f'Mean={np.mean(frames_dx):.2f}, Std={np.std(frames_dx):.2f}')
    axes[0].set_ylabel('X Offset (px)', fontsize=11)
    axes[0].set_title(f'X Offset Over Frames — Sequence: {longest_seq}', fontsize=12)
    axes[0].legend()
    
    axes[1].plot(frames_dy, 'r-', alpha=0.6, linewidth=0.8)
    axes[1].axhline(np.mean(frames_dy), color='red', linestyle='--', linewidth=2,
                    label=f'Mean={np.mean(frames_dy):.2f}, Std={np.std(frames_dy):.2f}')
    axes[1].set_ylabel('Y Offset (px)', fontsize=11)
    axes[1].set_xlabel('Frame Index', fontsize=11)
    axes[1].set_title(f'Y Offset Over Frames — Sequence: {longest_seq}', fontsize=12)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'offset_over_time.png'), dpi=150)
    plt.close()
    print(f"  Saved: offset_over_time.png")

print("\n" + "=" * 70)
print("INTERPRETATION GUIDE")
print("=" * 70)
print("""
If the offset is SENSOR-DEPENDENT (good for fixed calibration):
  → Histograms show a tight, single peak
  → Std is small relative to mean
  → 2D scatter is a tight cluster
  → Per-sequence offsets are similar across sequences
  → Over-time plot shows near-constant line

If the offset is SCENE-DEPENDENT (bad — needs per-frame alignment):
  → Histograms show wide spread or multiple peaks
  → Std is large relative to mean
  → 2D scatter is widely dispersed
  → Per-sequence offsets vary significantly
  → Over-time plot shows large fluctuations
""")

print(f"\nAll results saved to: {OUTPUT_DIR}")
