"""
Full alignment residual analysis for both train and val splits.
Saves per-sample data (CSV), summary statistics (JSON), and publication-ready context.
"""

import os
import glob
import json
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path
from collections import defaultdict

# Image resolutions (original, before any resizing)
RGB_W, RGB_H = 1920, 1080
IR_W, IR_H = 640, 512

BASE = r"D:\erkaki\yolov8_nano_ssd\data\anti-uav"
OUTPUT_DIR = r"D:\erkaki\yolov5_backup\alignment_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLITS = {
    "train": os.path.join(BASE, "labels", "train"),
    "val":   os.path.join(BASE, "labels", "val"),
}


def parse_label(filepath):
    labels = []
    if not os.path.exists(filepath):
        return labels
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                labels.append((int(parts[0]), float(parts[1]), float(parts[2]),
                               float(parts[3]), float(parts[4])))
    return labels


def extract_base_and_frame(filename):
    stem = Path(filename).stem
    if '_visible_' in stem:
        parts = stem.split('_visible_')
        return parts[0], int(parts[1]), 'visible'
    elif '_infrared_' in stem:
        parts = stem.split('_infrared_')
        return parts[0], int(parts[1]), 'infrared'
    return None, None, None


def analyze_split(split_name, label_dir):
    print(f"\n{'='*60}")
    print(f"  Analyzing: {split_name}")
    print(f"  Label dir: {label_dir}")
    print(f"{'='*60}")

    all_labels = glob.glob(os.path.join(label_dir, "*.txt"))
    rgb_labels = {}
    ir_labels = {}

    for lbl_path in all_labels:
        fname = os.path.basename(lbl_path)
        base, frame, modality = extract_base_and_frame(fname)
        if base is None:
            continue
        if modality == 'visible':
            rgb_labels[(base, frame)] = lbl_path
        elif modality == 'infrared':
            ir_labels[(base, frame)] = lbl_path

    matched_keys = sorted(set(rgb_labels.keys()) & set(ir_labels.keys()))
    print(f"  RGB labels: {len(rgb_labels)}")
    print(f"  IR labels:  {len(ir_labels)}")
    print(f"  Matched pairs: {len(matched_keys)}")

    rows = []
    skipped_empty = 0
    processed = 0

    for key in matched_keys:
        base, frame = key
        rgb_lbl = parse_label(rgb_labels[key])
        ir_lbl = parse_label(ir_labels[key])

        if len(rgb_lbl) == 0 or len(ir_lbl) == 0:
            skipped_empty += 1
            continue

        _, rgb_xc, rgb_yc, rgb_w, rgb_h = rgb_lbl[0]
        _, ir_xc, ir_yc, ir_w, ir_h = ir_lbl[0]

        # Normalized offsets
        dx_norm = ir_xc - rgb_xc
        dy_norm = ir_yc - rgb_yc

        # Pixel offsets in each image's native space
        rgb_xc_px = rgb_xc * RGB_W
        rgb_yc_px = rgb_yc * RGB_H
        ir_xc_px = ir_xc * IR_W
        ir_yc_px = ir_yc * IR_H

        # Scale RGB pixel coords into IR pixel space for fair comparison
        rgb_xc_in_ir = rgb_xc_px * (IR_W / RGB_W)
        rgb_yc_in_ir = rgb_yc_px * (IR_H / RGB_H)

        dx_px = ir_xc_px - rgb_xc_in_ir
        dy_px = ir_yc_px - rgb_yc_in_ir

        # BBox sizes (in native pixels)
        rgb_bbox_w_px = rgb_w * RGB_W
        rgb_bbox_h_px = rgb_h * RGB_H
        ir_bbox_w_px = ir_w * IR_W
        ir_bbox_h_px = ir_h * IR_H

        rows.append({
            'split': split_name,
            'sequence': base,
            'frame': frame,
            'rgb_xc_norm': rgb_xc, 'rgb_yc_norm': rgb_yc,
            'rgb_w_norm': rgb_w, 'rgb_h_norm': rgb_h,
            'ir_xc_norm': ir_xc, 'ir_yc_norm': ir_yc,
            'ir_w_norm': ir_w, 'ir_h_norm': ir_h,
            'dx_norm': dx_norm, 'dy_norm': dy_norm,
            'rgb_xc_px': rgb_xc_px, 'rgb_yc_px': rgb_yc_px,
            'ir_xc_px': ir_xc_px, 'ir_yc_px': ir_yc_px,
            'dx_ir_px': dx_px, 'dy_ir_px': dy_px,
            'rgb_bbox_w_px': rgb_bbox_w_px, 'rgb_bbox_h_px': rgb_bbox_h_px,
            'ir_bbox_w_px': ir_bbox_w_px, 'ir_bbox_h_px': ir_bbox_h_px,
        })
        processed += 1

    print(f"  Processed: {processed}, Skipped (empty): {skipped_empty}")
    return rows


def compute_stats(rows):
    dx = np.array([r['dx_ir_px'] for r in rows])
    dy = np.array([r['dy_ir_px'] for r in rows])
    dx_n = np.array([r['dx_norm'] for r in rows])
    dy_n = np.array([r['dy_norm'] for r in rows])

    displacement = np.sqrt(dx**2 + dy**2)

    stats = {
        'n_samples': len(rows),
        'pixel_offsets_ir_space': {
            'dx': {'mean': float(dx.mean()), 'std': float(dx.std()),
                   'min': float(dx.min()), 'max': float(dx.max()),
                   'median': float(np.median(dx)),
                   'q25': float(np.percentile(dx, 25)),
                   'q75': float(np.percentile(dx, 75))},
            'dy': {'mean': float(dy.mean()), 'std': float(dy.std()),
                   'min': float(dy.min()), 'max': float(dy.max()),
                   'median': float(np.median(dy)),
                   'q25': float(np.percentile(dy, 25)),
                   'q75': float(np.percentile(dy, 75))},
            'euclidean_displacement': {
                'mean': float(displacement.mean()),
                'std': float(displacement.std()),
                'median': float(np.median(displacement)),
                'min': float(displacement.min()),
                'max': float(displacement.max()),
            },
            'cv_dx': float(dx.std() / abs(dx.mean())) if abs(dx.mean()) > 1e-6 else float('inf'),
            'cv_dy': float(dy.std() / abs(dy.mean())) if abs(dy.mean()) > 1e-6 else float('inf'),
        },
        'normalized_offsets': {
            'dx': {'mean': float(dx_n.mean()), 'std': float(dx_n.std()),
                   'min': float(dx_n.min()), 'max': float(dx_n.max())},
            'dy': {'mean': float(dy_n.mean()), 'std': float(dy_n.std()),
                   'min': float(dy_n.min()), 'max': float(dy_n.max())},
        },
        'within_threshold_pct': {},
    }

    for t in [2, 5, 10, 15, 20, 30, 50]:
        within = np.abs(dx - dx.mean()) < t
        within_y = np.abs(dy - dy.mean()) < t
        both = within & within_y
        stats['within_threshold_pct'][f'pm_{t}px'] = {
            'dx_only': float(100.0 * within.sum() / len(within)),
            'dy_only': float(100.0 * within_y.sum() / len(within_y)),
            'both': float(100.0 * both.sum() / len(both)),
        }

    # Per-sequence stats
    seqs = defaultdict(lambda: {'dx': [], 'dy': []})
    for r in rows:
        seqs[r['sequence']]['dx'].append(r['dx_ir_px'])
        seqs[r['sequence']]['dy'].append(r['dy_ir_px'])

    seq_stats = {}
    for s in sorted(seqs.keys()):
        sdx = np.array(seqs[s]['dx'])
        sdy = np.array(seqs[s]['dy'])
        seq_stats[s] = {
            'n_frames': len(sdx),
            'dx_mean': float(sdx.mean()), 'dx_std': float(sdx.std()),
            'dy_mean': float(sdy.mean()), 'dy_std': float(sdy.std()),
        }
    stats['per_sequence'] = seq_stats

    # Variance decomposition: between-sequence vs within-sequence
    seq_means_dx = np.array([seq_stats[s]['dx_mean'] for s in seq_stats])
    seq_means_dy = np.array([seq_stats[s]['dy_mean'] for s in seq_stats])
    seq_counts = np.array([seq_stats[s]['n_frames'] for s in seq_stats])

    grand_mean_dx = dx.mean()
    grand_mean_dy = dy.mean()

    var_between_dx = float(np.average((seq_means_dx - grand_mean_dx)**2, weights=seq_counts))
    var_within_dx = float(np.average(
        [seq_stats[s]['dx_std']**2 for s in seq_stats], weights=seq_counts))
    var_total_dx = float(dx.var())

    var_between_dy = float(np.average((seq_means_dy - grand_mean_dy)**2, weights=seq_counts))
    var_within_dy = float(np.average(
        [seq_stats[s]['dy_std']**2 for s in seq_stats], weights=seq_counts))
    var_total_dy = float(dy.var())

    stats['variance_decomposition'] = {
        'dx': {
            'total_var': var_total_dx,
            'between_seq_var': var_between_dx,
            'within_seq_var': var_within_dx,
            'between_seq_pct': float(100 * var_between_dx / var_total_dx) if var_total_dx > 0 else 0,
            'within_seq_pct': float(100 * var_within_dx / var_total_dx) if var_total_dx > 0 else 0,
        },
        'dy': {
            'total_var': var_total_dy,
            'between_seq_var': var_between_dy,
            'within_seq_var': var_within_dy,
            'between_seq_pct': float(100 * var_between_dy / var_total_dy) if var_total_dy > 0 else 0,
            'within_seq_pct': float(100 * var_within_dy / var_total_dy) if var_total_dy > 0 else 0,
        },
    }

    return stats


def save_csv(rows, filepath):
    if not rows:
        return
    keys = rows[0].keys()
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved CSV: {filepath} ({len(rows)} rows)")


def save_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"  Saved JSON: {filepath}")


# ────────────────────────────────────────────────────────────────
#  MAIN
# ────────────────────────────────────────────────────────────────
all_results = {}

for split_name, label_dir in SPLITS.items():
    rows = analyze_split(split_name, label_dir)

    # Save per-sample CSV
    csv_path = os.path.join(OUTPUT_DIR, f"alignment_residuals_{split_name}.csv")
    save_csv(rows, csv_path)

    # Compute and save summary stats
    if rows:
        stats = compute_stats(rows)
        stats['split'] = split_name
        stats['image_resolutions'] = {
            'rgb': {'width': RGB_W, 'height': RGB_H},
            'ir': {'width': IR_W, 'height': IR_H},
        }
        json_path = os.path.join(OUTPUT_DIR, f"alignment_stats_{split_name}.json")
        save_json(stats, json_path)
        all_results[split_name] = {'rows': rows, 'stats': stats}

        # Print key results
        px = stats['pixel_offsets_ir_space']
        vd = stats['variance_decomposition']
        print(f"\n  --- {split_name} Summary (IR pixel space) ---")
        print(f"  dx: mean={px['dx']['mean']:.2f}, std={px['dx']['std']:.2f}, "
              f"median={px['dx']['median']:.2f}, CV={px['cv_dx']:.2f}")
        print(f"  dy: mean={px['dy']['mean']:.2f}, std={px['dy']['std']:.2f}, "
              f"median={px['dy']['median']:.2f}, CV={px['cv_dy']:.2f}")
        print(f"  Euclidean: mean={px['euclidean_displacement']['mean']:.2f}, "
              f"std={px['euclidean_displacement']['std']:.2f}")
        print(f"  Variance decomposition (dx): "
              f"between-seq={vd['dx']['between_seq_pct']:.1f}%, "
              f"within-seq={vd['dx']['within_seq_pct']:.1f}%")
        print(f"  Variance decomposition (dy): "
              f"between-seq={vd['dy']['between_seq_pct']:.1f}%, "
              f"within-seq={vd['dy']['within_seq_pct']:.1f}%")

# Save combined comparison JSON
if len(all_results) == 2:
    comparison = {}
    for split in ['train', 'val']:
        s = all_results[split]['stats']
        comparison[split] = {
            'n_samples': s['n_samples'],
            'n_sequences': len(s['per_sequence']),
            'dx_mean': s['pixel_offsets_ir_space']['dx']['mean'],
            'dx_std': s['pixel_offsets_ir_space']['dx']['std'],
            'dx_median': s['pixel_offsets_ir_space']['dx']['median'],
            'dx_cv': s['pixel_offsets_ir_space']['cv_dx'],
            'dy_mean': s['pixel_offsets_ir_space']['dy']['mean'],
            'dy_std': s['pixel_offsets_ir_space']['dy']['std'],
            'dy_median': s['pixel_offsets_ir_space']['dy']['median'],
            'dy_cv': s['pixel_offsets_ir_space']['cv_dy'],
            'euclidean_mean': s['pixel_offsets_ir_space']['euclidean_displacement']['mean'],
            'euclidean_std': s['pixel_offsets_ir_space']['euclidean_displacement']['std'],
            'var_between_dx_pct': s['variance_decomposition']['dx']['between_seq_pct'],
            'var_within_dx_pct': s['variance_decomposition']['dx']['within_seq_pct'],
            'var_between_dy_pct': s['variance_decomposition']['dy']['between_seq_pct'],
            'var_within_dy_pct': s['variance_decomposition']['dy']['within_seq_pct'],
        }
    save_json(comparison, os.path.join(OUTPUT_DIR, "alignment_comparison_train_val.json"))

# ────────────────────────────────────────────────────────────────
#  Generate publication-quality figures
# ────────────────────────────────────────────────────────────────
print("\n\nGenerating publication figures...")

for split_name in all_results:
    rows = all_results[split_name]['rows']
    stats = all_results[split_name]['stats']
    dx = np.array([r['dx_ir_px'] for r in rows])
    dy = np.array([r['dy_ir_px'] for r in rows])

    # --- Figure 1: Histograms ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(dx, bins=80, color='#4878CF', edgecolor='black', alpha=0.85, linewidth=0.5)
    axes[0].axvline(dx.mean(), color='#E24A33', linewidth=2, linestyle='--',
                    label=f'Mean = {dx.mean():.1f} px')
    axes[0].axvline(dx.mean() - dx.std(), color='#FFA500', linewidth=1.5, linestyle=':',
                    label=f'Std = {dx.std():.1f} px')
    axes[0].axvline(dx.mean() + dx.std(), color='#FFA500', linewidth=1.5, linestyle=':')
    axes[0].set_xlabel('Horizontal Offset $\\Delta x$ (pixels)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title(f'(a) Horizontal offset ({split_name})', fontsize=12)
    axes[0].legend(fontsize=10)

    axes[1].hist(dy, bins=80, color='#E24A33', edgecolor='black', alpha=0.85, linewidth=0.5)
    axes[1].axvline(dy.mean(), color='#E24A33', linewidth=2, linestyle='--',
                    label=f'Mean = {dy.mean():.1f} px')
    axes[1].axvline(dy.mean() - dy.std(), color='#FFA500', linewidth=1.5, linestyle=':',
                    label=f'Std = {dy.std():.1f} px')
    axes[1].axvline(dy.mean() + dy.std(), color='#FFA500', linewidth=1.5, linestyle=':')
    axes[1].set_xlabel('Vertical Offset $\\Delta y$ (pixels)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title(f'(b) Vertical offset ({split_name})', fontsize=12)
    axes[1].legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'histograms_{split_name}.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'histograms_{split_name}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved histograms_{split_name}.png/pdf")

    # --- Figure 2: 2D scatter ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(dx, dy, alpha=0.12, s=6, c='#4878CF', rasterized=True)
    ax.scatter([dx.mean()], [dy.mean()], color='#E24A33', s=200, marker='+',
               linewidths=3, zorder=5,
               label=f'Mean ({dx.mean():.1f}, {dy.mean():.1f})')

    for n_std in [1, 2, 3]:
        e = Ellipse(xy=(dx.mean(), dy.mean()),
                    width=2*n_std*dx.std(), height=2*n_std*dy.std(),
                    fill=False, edgecolor='#E24A33', linestyle='--', linewidth=1.5,
                    label=f'{n_std}$\\sigma$')
        ax.add_patch(e)

    ax.set_xlabel('Horizontal Offset $\\Delta x$ (pixels)', fontsize=12)
    ax.set_ylabel('Vertical Offset $\\Delta y$ (pixels)', fontsize=12)
    ax.set_title(f'Joint Offset Distribution ({split_name})', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'scatter2d_{split_name}.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'scatter2d_{split_name}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved scatter2d_{split_name}.png/pdf")

    # --- Figure 3: Per-sequence boxplot ---
    seqs = defaultdict(lambda: {'dx': [], 'dy': []})
    for r in rows:
        seqs[r['sequence']]['dx'].append(r['dx_ir_px'])
        seqs[r['sequence']]['dy'].append(r['dy_ir_px'])

    seq_names = sorted(seqs.keys())
    dx_per_seq = [seqs[s]['dx'] for s in seq_names]
    dy_per_seq = [seqs[s]['dy'] for s in seq_names]
    short_names = [s.split('_')[-1] for s in seq_names]

    fig, axes = plt.subplots(2, 1, figsize=(max(10, len(seq_names)*0.8), 7))

    bp1 = axes[0].boxplot(dx_per_seq, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='#4878CF', alpha=0.6),
                          medianprops=dict(color='black', linewidth=1.5))
    axes[0].set_ylabel('$\\Delta x$ (px)', fontsize=11)
    axes[0].set_title(f'Per-Sequence Horizontal Offset ({split_name})', fontsize=12)
    axes[0].axhline(dx.mean(), color='#E24A33', linestyle='--', alpha=0.7, label='Global mean')
    axes[0].set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    axes[0].legend(fontsize=9)

    bp2 = axes[1].boxplot(dy_per_seq, vert=True, patch_artist=True,
                          boxprops=dict(facecolor='#E24A33', alpha=0.6),
                          medianprops=dict(color='black', linewidth=1.5))
    axes[1].set_ylabel('$\\Delta y$ (px)', fontsize=11)
    axes[1].set_title(f'Per-Sequence Vertical Offset ({split_name})', fontsize=12)
    axes[1].axhline(dy.mean(), color='#E24A33', linestyle='--', alpha=0.7, label='Global mean')
    axes[1].set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'per_sequence_{split_name}.png'), dpi=200, bbox_inches='tight')
    plt.savefig(os.path.join(OUTPUT_DIR, f'per_sequence_{split_name}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved per_sequence_{split_name}.png/pdf")


# ────────────────────────────────────────────────────────────────
#  Save context file for AI visualization
# ────────────────────────────────────────────────────────────────
context = """
ALIGNMENT RESIDUAL ANALYSIS - CONTEXT FOR VISUALIZATION / TABLE GENERATION
===========================================================================

PURPOSE:
--------
This analysis quantifies the spatial displacement (offset) between RGB and
infrared (IR) object annotations in the Anti-UAV dataset. The goal is to
determine whether the inter-modal offset is a fixed sensor-dependent constant
or a variable quantity that changes per sample / per sequence.

DATASET:
--------
- Anti-UAV dataset for drone detection
- Single class: drone/UAV
- RGB resolution: 1920x1080 pixels
- IR resolution: 640x512 pixels
- Cameras are rigidly co-mounted on the same platform
- Each frame has separate YOLO-format labels for RGB and IR
  (class x_center y_center width height, normalized 0-1)

METHODOLOGY:
------------
For each matched RGB-IR label pair:
  1. Read normalized center coordinates from both labels
  2. Convert to pixel coordinates in each image's native resolution
  3. Project RGB pixel coords into IR pixel space: rgb_x_in_ir = rgb_x_px * (IR_W / RGB_W)
  4. Compute offset: dx = ir_x_px - rgb_x_in_ir, dy = ir_y_px - rgb_y_in_ir
  5. This gives the displacement in IR pixel space

FILES PRODUCED:
---------------
Per-split CSV files (alignment_residuals_train.csv, alignment_residuals_val.csv):
  Columns:
    - split: train or val
    - sequence: video sequence identifier
    - frame: frame number within sequence
    - rgb_xc_norm, rgb_yc_norm, rgb_w_norm, rgb_h_norm: RGB label (normalized)
    - ir_xc_norm, ir_yc_norm, ir_w_norm, ir_h_norm: IR label (normalized)
    - dx_norm, dy_norm: offset in normalized coordinates
    - rgb_xc_px, rgb_yc_px: RGB center in native pixels
    - ir_xc_px, ir_yc_px: IR center in native pixels
    - dx_ir_px, dy_ir_px: offset in IR pixel space (THE KEY COLUMNS)
    - rgb_bbox_w_px, rgb_bbox_h_px: RGB bbox size in native pixels
    - ir_bbox_w_px, ir_bbox_h_px: IR bbox size in native pixels

Per-split JSON files (alignment_stats_train.json, alignment_stats_val.json):
  Contains: mean, std, min, max, median, percentiles, CV, variance decomposition,
  per-sequence stats, threshold analysis

Comparison JSON (alignment_comparison_train_val.json):
  Side-by-side comparison of key metrics for train vs val

WHAT TO REPORT IN THE PAPER:
-----------------------------
1. TABLE (recommended format):
   A compact table comparing train/val statistics:

   | Metric                    | Train          | Val            |
   |---------------------------|----------------|----------------|
   | N samples                 | ...            | ...            |
   | N sequences               | ...            | ...            |
   | dx mean (px)              | ...            | ...            |
   | dx std (px)               | ...            | ...            |
   | dx CV                     | ...            | ...            |
   | dy mean (px)              | ...            | ...            |
   | dy std (px)               | ...            | ...            |
   | dy CV                     | ...            | ...            |
   | Euclidean mean (px)       | ...            | ...            |
   | Between-seq variance (dx) | ...%           | ...%           |
   | Within-seq variance (dx)  | ...%           | ...%           |

2. FIGURE (recommended: 3-panel figure):
   (a) dx histogram (use train or combined)
   (b) dy histogram
   (c) 2D scatter with sigma ellipses
   Optional 4th panel: per-sequence boxplot

3. KEY NUMBERS TO CITE IN TEXT:
   - Coefficient of Variation (CV): proves offset is NOT constant
     (CV >> 1 means std >> mean, i.e. highly variable)
   - Variance decomposition: what % is between-sequence vs within-sequence
   - % of samples within +/-N pixels of mean

4. INTERPRETATION:
   - If CV is high and between-sequence variance dominates:
     offset depends on recording setup (different sensor configurations)
   - If within-sequence variance is also significant:
     offset also depends on object position/depth (parallax)
   - Either way: a single fixed offset is insufficient,
     justifying per-sample unified labeling for training

KEY COLUMNS IN CSV FOR PLOTTING:
  - dx_ir_px, dy_ir_px: for histograms and scatter plots
  - sequence: for per-sequence grouping / boxplots
  - frame: for temporal plots within a sequence
"""

context_path = os.path.join(OUTPUT_DIR, "CONTEXT_README.txt")
with open(context_path, 'w') as f:
    f.write(context)
print(f"\nSaved context file: {context_path}")

print("\n" + "="*60)
print("  ALL DONE")
print("="*60)
print(f"Output directory: {OUTPUT_DIR}")
print("Files:")
for fname in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, fname)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {fname:45s} {size_kb:8.1f} KB")
