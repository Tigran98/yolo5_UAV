
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
