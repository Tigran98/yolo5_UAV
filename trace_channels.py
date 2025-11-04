"""Trace channel values during model building"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from copy import deepcopy
from models.yolo import parse_model

# Load config
with open('models/yolov5n_fusion.yaml') as f:
    cfg = yaml.safe_load(f)

# Simulate the fusion model's head building
backbone_len = 10
head_cfg = deepcopy(cfg)
head_cfg["backbone"] = []

# Initialize ch_list as fusion model does
fusion_indices = [4, 6, 9]
fusion_channels = [64, 128, 256]
backbone_max_idx = max(fusion_indices)
ch_list = [None] * (backbone_max_idx + 1)
for idx, c in zip(fusion_indices, fusion_channels):
    ch_list[idx] = c

# Extend for absolute indices
max_ref_idx = max(fusion_indices)
for layer in head_cfg["head"]:
    f = layer[0]
    if isinstance(f, list):
        max_ref_idx = max(max_ref_idx, max(x for x in f if x >= 0))
    elif isinstance(f, int) and f >= 0:
        max_ref_idx = max(max_ref_idx, f)

while len(ch_list) <= max_ref_idx:
    ch_list.append(None)

# Create ChList wrapper
class ChList(list):
    def __init__(self, iterable):
        super().__init__(iterable)
    
    def __getitem__(self, idx):
        if isinstance(idx, int) and idx < 0:
            val = list.__getitem__(self, idx)
            if val is None:
                for i in range(len(self) - 1, -1, -1):
                    if self[i] is not None:
                        return list.__getitem__(self, i)
            return val
        elif isinstance(idx, int) and idx >= len(self):
            while len(self) <= idx:
                self.append(None)
        return list.__getitem__(self, idx)

ch = ChList(ch_list)

print("Initial ch:", [ch[i] if i < len(ch) else None for i in range(25)])
print()

# Trace through all head layers, but focus on 4, 8, 9, 10
for i, (f, n, m, args) in enumerate(head_cfg["head"]):
    if i not in [4, 8, 9, 10]:
        # Still process but don't print details
        if m == "Concat":
            channel_values = []
            for x in f:
                if x == -1:
                    val = ch[-1]
                else:
                    val = ch[x] if x < len(ch) else None
                if val is not None:
                    channel_values.append(val)
            c2 = sum(channel_values)
        else:
            if f == -1:
                c1 = ch[-1]
            else:
                c1 = ch[f] if f < len(ch) else None
            c2 = args[0] if isinstance(args, list) and len(args) > 0 else args
            if isinstance(c2, (int, float)):
                gw = cfg['width_multiple']
                c2 = int(c2 * gw)
            else:
                c2 = 0  # Placeholder
        
        ch.append(c2)
        abs_idx = backbone_len + i
        while len(ch) <= abs_idx:
            ch.append(None)
        ch[abs_idx] = c2
        continue
    
    print(f"Layer {i} (abs={backbone_len+i}): f={f}, args={args}")
    print(f"  ch[-1] = {ch[-1]}")
    print(f"  ch length = {len(ch)}")
    
    if m == "Concat":
        # Sum channels
        channel_values = []
        for x in f:
            if x == -1:
                val = ch[-1]
                print(f"    x={x} (-1): ch[-1]={val}")
            else:
                val = ch[x] if x < len(ch) else None
                print(f"    x={x}: ch[{x}]={val}, ch length={len(ch)}")
            if val is not None:
                channel_values.append(val)
        c2 = sum(channel_values)
        print(f"  Concat: {channel_values} -> c2={c2}")
    else:
        # Get c1 from f
        if f == -1:
            c1 = ch[-1]
        else:
            c1 = ch[f]
        c2 = args[0]
        gw = cfg['width_multiple']
        c2 = int(c2 * gw)
        print(f"  c1={c1}, c2={c2} (from args[0]={args[0]}, gw={gw})")
    
    # Append
    ch.append(c2)
    abs_idx = backbone_len + i
    while len(ch) <= abs_idx:
        ch.append(None)
    ch[abs_idx] = c2
    
    print(f"  After append: ch[-1]={ch[-1]}, ch[{abs_idx}]={ch[abs_idx]}")
    print(f"  ch length = {len(ch)}")
    print()
    
    if i >= 10:
        break

