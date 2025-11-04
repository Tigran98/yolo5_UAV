"""Check actual backbone output channels"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from models.yolo import parse_model
from copy import deepcopy
import yaml
import torch

# Load config
with open('models/yolov5n_fusion.yaml') as f:
    cfg = yaml.safe_load(f)

# Build backbone
cfg['head'] = []
backbone, _ = parse_model(deepcopy(cfg), ch=[3])

# Get actual output channels at fusion indices
fusion_indices = [4, 6, 9]
actual_channels = []

# Create dummy input
x = torch.randn(1, 3, 256, 256)

for i, layer in enumerate(backbone):
    x = layer(x)
    if i in fusion_indices:
        ch = x.shape[1]
        actual_channels.append(ch)
        print(f"Layer {i}: output channels = {ch}, shape = {x.shape}")

print(f"\nActual fusion channels: {actual_channels}")
print(f"Config fusion_channels (before width_multiple): {cfg['fusion_channels']}")
gw = cfg['width_multiple']
config_channels_scaled = [int(c * gw) for c in cfg['fusion_channels']]
print(f"Config fusion_channels (after width_multiple={gw}): {config_channels_scaled}")


