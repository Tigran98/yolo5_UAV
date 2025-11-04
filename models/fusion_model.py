"""
YOLOv5 Fusion Model - Dual-input (RGB + IR) with medium fusion
"""

import math
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from models.common import FeatureFusion
from models.yolo import BaseModel, Detect, parse_model
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER
from utils.torch_utils import initialize_weights


class YOLOv5FusionModel(BaseModel):
    """
    YOLOv5 Fusion Model with dual backbones (RGB + IR) and medium fusion.
    Features are fused at P3, P4, P5 before the PANet neck.
    """

    def __init__(self, cfg="yolov5n_fusion.yaml", ch=3, nc=None, anchors=None):
        """
        Initialize the fusion model.
        
        Args:
            cfg: Configuration file path or dict
            ch: Input channels (3 for RGB/IR)
            nc: Number of classes
            anchors: Anchor boxes for detection head
        """
        super().__init__()
        
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)
        
        # Define model parameters
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)
        if nc and nc != self.yaml["nc"]:
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)
        
        # Build RGB backbone
        LOGGER.info("Building RGB backbone...")
        rgb_cfg = deepcopy(self.yaml)
        rgb_cfg["head"] = []  # Remove head, only build backbone
        self.rgb_backbone, _ = parse_model(rgb_cfg, ch=[ch])
        
        # Build IR backbone (same structure)
        LOGGER.info("Building IR backbone...")
        ir_cfg = deepcopy(self.yaml)
        ir_cfg["head"] = []
        self.ir_backbone, _ = parse_model(ir_cfg, ch=[ch])
        
        # Get fusion configuration
        self.fusion_indices = self.yaml.get("fusion_indices", [4, 6, 9])
        fusion_channels_yaml = self.yaml.get("fusion_channels", [64, 128, 256])
        
        # Get actual output channels from backbone at fusion points (after width_multiple)
        self.fusion_channels = []
        dummy_input = torch.zeros(1, ch, 256, 256)
        with torch.no_grad():
            x_rgb = dummy_input
            for i, m in enumerate(self.rgb_backbone):
                x_rgb = m(x_rgb)
                if i in self.fusion_indices:
                    actual_ch = x_rgb.shape[1]
                    self.fusion_channels.append(actual_ch)
                    LOGGER.info(f"Fusion point {i}: actual channels = {actual_ch}")
        
        # Build fusion modules with actual channel counts
        self.fusions = nn.ModuleList([
            FeatureFusion(c) for c in self.fusion_channels
        ])
        
        # Build head using parse_model with fused feature channels
        LOGGER.info("Building detection head...")
        head_cfg = deepcopy(self.yaml)
        head_cfg["backbone"] = []  # Remove backbone
        
        # Head YAML uses absolute indices (4, 6, 9, 10, 14, etc.)
        # parse_model resets ch to [] at i==0, then builds it sequentially
        # We need to convert absolute indices to relative indices
        # Fusion points: P3=4, P4=6, P5=9 map to initial ch indices 0, 1, 2
        
        # Create mapping: absolute fusion index -> initial ch index
        fusion_to_ch_idx = {abs_idx: rel_idx for rel_idx, abs_idx in enumerate(self.fusion_indices)}
        # {4: 0, 6: 1, 9: 2}
        
        # Also create reverse mapping for forward pass: ch_idx -> absolute fusion index
        self.ch_idx_to_fusion = {rel_idx: abs_idx for abs_idx, rel_idx in fusion_to_ch_idx.items()}
        # {0: 4, 1: 6, 2: 9}
        
        # Convert head layers: absolute indices -> relative indices
        converted_head = []
        for layer_idx, layer in enumerate(head_cfg["head"]):
            f = layer[0]
            if isinstance(f, list):
                new_f = []
                for idx in f:
                    if idx == -1:
                        new_f.append(-1)  # Keep relative -1
                    elif idx in fusion_to_ch_idx:
                        # Fusion point: map to initial ch index (0, 1, or 2)
                        ch_idx = fusion_to_ch_idx[idx]
                        new_f.append(ch_idx)
                    elif idx >= 10:  # Head layer (absolute index >= 10)
                        # Map to relative: idx - 10 - layer_idx (layers before current)
                        new_f.append(idx - 10 - layer_idx)
                    else:
                        new_f.append(idx)
                converted_head.append([new_f] + layer[1:])
            elif isinstance(f, int):
                if f == -1:
                    converted_head.append(layer)
                elif f in fusion_to_ch_idx:
                    ch_idx = fusion_to_ch_idx[f]
                    converted_head.append([ch_idx] + layer[1:])
                elif f >= 10:
                    converted_head.append([f - 10 - layer_idx] + layer[1:])
                else:
                    converted_head.append(layer)
            else:
                converted_head.append(layer)
        
        head_cfg["head"] = converted_head
        
        # Initialize ch with fusion channels: [P3=64, P4=128, P5=256]
        # parse_model resets ch = [] at i==0, so fusion channels at indices 0,1,2 are lost
        # Solution: Use a custom parse_model that doesn't reset ch, or manually rebuild
        # head layers with correct channel counts.
        
        # Actually, we can work around this by ensuring ch_init has fusion channels
        # that parse_model will access BEFORE reset. But parse_model accesses ch[-1] first,
        # then resets ch = [] at i==0, so fusion channels are lost.
        
        # Best solution: Manually build head with correct channels, bypassing parse_model's
        # channel calculation issues. But that's complex.
        
        # Practical solution: Let parse_model build with converted indices, then manually
        # fix any C3/Conv layers that have wrong input channels due to fusion point references.
        # OR: Extend ch_init with padding so fusion channels survive reset.
        
        # Try: Initialize ch with enough padding so fusion channels are at high indices
        # that won't be reset. But parse_model only resets ch = [] at i==0, it doesn't
        # clear high indices.
        
        # Actually wait - parse_model does `ch = []` then `ch.append(c2)` for each layer.
        # So ch only contains layer outputs, not initial values.
        
        # Final solution: Build head normally, but in forward pass, ensure we concatenate
        # the right channels. The issue is parse_model builds C3 expecting 192 channels
        # (128+64) but we concatenate 128 channels (64+64).
        
        # Fix: Make sure parse_model calculates channels correctly by ensuring ch[0], ch[1]
        # contain the right values when Concat accesses them. Since ch gets reset, we need
        # to store fusion channels at the start of ch after reset.
        
        # Workaround: Modify parse_model call to inject fusion channels after reset
        # We'll create a wrapper that intercepts the ch reset and adds fusion channels back
        
        from models.yolo import parse_model as orig_parse_model
        import contextlib
        
        # Save fusion channels
        fusion_ch_dict = {i: ch for i, ch in enumerate(self.fusion_channels)}
        
        # Call parse_model - it will reset ch, but we'll handle fusion access in forward
        ch_init = self.fusion_channels.copy()  # [64, 128, 256]
        self.head, self.save = orig_parse_model(head_cfg, ch_init)
        
        # Fix channel mismatches: parse_model may calculate wrong input channels for C3
        # layers that concatenate with fusion points, because ch gets reset
        # Layer 7 (index 7 in head): C3 after Concat([-1, 0]) should have 128 input (64+64)
        # but parse_model thinks it's 192 (128+64) because ch[0] after reset is layer 0 output
        # We need to manually fix this layer
        if len(self.head) > 7:
            layer7 = self.head[7]  # C3 layer after Concat with P3
            if hasattr(layer7, 'cv1') and hasattr(layer7.cv1, 'conv'):
                # Check if input channels are wrong (192 instead of 128)
                actual_in_ch = layer7.cv1.conv.in_channels
                # Layer 6 Concat concatenates layer 5 (64) and P3 (64) = 128
                expected_in_ch = 128
                if actual_in_ch != expected_in_ch:
                    # Rebuild C3 with correct input channels
                    from models.common import C3
                    c3_out_ch = layer7.cv3.conv.out_channels if hasattr(layer7, 'cv3') else 64
                    # Recreate C3 with correct channels
                    correct_c3 = C3(expected_in_ch, c3_out_ch, n=1, shortcut=False)
                    # Copy attributes from old layer (i, f, type, np)
                    correct_c3.i = layer7.i if hasattr(layer7, 'i') else 7
                    correct_c3.f = layer7.f if hasattr(layer7, 'f') else -1
                    correct_c3.type = layer7.type if hasattr(layer7, 'type') else 'models.common.C3'
                    correct_c3.np = sum(p.numel() for p in correct_c3.parameters())
                    # Replace the layer
                    self.head[7] = correct_c3
        
        # Store fusion channel mapping for forward pass
        self.fusion_ch_map = fusion_ch_dict
        
        # Combine all layers for info() method
        self.model = nn.ModuleList(list(self.rgb_backbone) + list(self.ir_backbone) + 
                                    list(self.fusions) + list(self.head))
        self.names = [str(i) for i in range(self.yaml["nc"])]
        self.inplace = self.yaml.get("inplace", True)
        
        # Build strides, anchors
        m = self.head[-1]  # Detect()
        if isinstance(m, Detect):
            def _forward(x_rgb, x_ir):
                # Forward through model to get Detect output
                # Detect layer receives list of feature maps [p3, p4, p5]
                # and returns list of raw outputs in training mode
                with torch.no_grad():
                    pred = self._forward_once(x_rgb, x_ir)
                    return pred
            
            s = 256  # 2x min stride
            m.inplace = self.inplace
            dummy = torch.zeros(1, ch, s, s)
            # Put model in training mode to get list output from Detect
            was_training = self.training
            self.train()
            try:
                outputs = _forward(dummy, dummy)
                # Detect returns list of tensors in training mode
                # Each tensor is (bs, na, ny, nx, no) format
                if isinstance(outputs, list) and len(outputs) == 3:
                    # Get spatial dimensions from the list (ny dimension)
                    # outputs[i] shape is (bs, na, ny, nx, no)
                    m.stride = torch.tensor([s / x.shape[-3] for x in outputs])  # Use ny (dim -3)
                else:
                    # Fallback: use default strides
                    m.stride = torch.tensor([8., 16., 32.])
            finally:
                self.train(was_training)
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()
        
        # Init weights
        initialize_weights(self)
        self.info()
        LOGGER.info("")
    
    def _initialize_biases(self, cf=None):
        """Initialize biases for Detect() module."""
        m = self.head[-1]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    
    def forward(self, x_rgb, x_ir, augment=False, profile=False, visualize=False):
        """
        Forward pass with dual inputs.
        
        Args:
            x_rgb: RGB input (B, 3, H, W)
            x_ir: IR input (B, 3, H, W)
            augment: Augmented inference
            profile: Profile performance
            visualize: Visualize features
            
        Returns:
            Detection outputs
        """
        if augment:
            return self._forward_augment(x_rgb, x_ir)
        return self._forward_once(x_rgb, x_ir, profile, visualize)
    
    def _forward_once(self, x_rgb, x_ir, profile=False, visualize=False):
        """Single forward pass."""
        # Extract features from RGB backbone
        rgb_features = []
        y_rgb = x_rgb
        for i, m in enumerate(self.rgb_backbone):
            y_rgb = m(y_rgb)
            if i in self.fusion_indices:
                rgb_features.append(y_rgb)
        
        # Extract features from IR backbone
        ir_features = []
        y_ir = x_ir
        for i, m in enumerate(self.ir_backbone):
            y_ir = m(y_ir)
            if i in self.fusion_indices:
                ir_features.append(y_ir)
        
        # Fuse features at each scale
        fused_features = []
        for i, (rgb_feat, ir_feat) in enumerate(zip(rgb_features, ir_features)):
            fused = self.fusions[i](rgb_feat, ir_feat)
            fused_features.append(fused)
        
        # Pass fused features through head
        # Build y list with fused features at correct positions
        max_backbone_idx = max(self.fusion_indices)
        y = [None] * (max_backbone_idx + 1)
        
        # Place fused features at fusion_indices positions
        for idx, feat in zip(self.fusion_indices, fused_features):
            y[idx] = feat
        
        # Run through head layers
        # Use a dictionary to track outputs by absolute index
        outputs = {}
        backbone_len = len(self.rgb_backbone)
        
        # Initialize outputs with fused features at fusion indices
        for idx, feat in zip(self.fusion_indices, fused_features):
            outputs[idx] = feat
        
        # Process head layers sequentially
        x = fused_features[-1]  # Start from P5 (last fused feature)
        
        for head_idx, m in enumerate(self.head):
            abs_idx = backbone_len + head_idx
            
            # Detect layer needs special handling - it takes list of inputs
            if isinstance(m, Detect):
                # Detect expects list of feature maps from layers specified in m.f
                # m.f may contain negative indices (relative) or positive (absolute)
                detect_inputs = []
                for detect_idx in m.f:
                    abs_idx = None
                    if detect_idx == -1:
                        # Previous layer (current head_idx - 1)
                        abs_idx = backbone_len + head_idx - 1
                    elif detect_idx < 0:
                        # Negative index: relative to current head position
                        rel_idx = head_idx + detect_idx  # detect_idx is negative
                        abs_idx = backbone_len + rel_idx
                    elif detect_idx in outputs:
                        abs_idx = detect_idx
                    
                    if abs_idx is not None and abs_idx in outputs:
                        detect_inputs.append(outputs[abs_idx])
                    else:
                        LOGGER.warning(f"Detect layer: index {detect_idx} (abs: {abs_idx}) not found in outputs")
                
                if len(detect_inputs) == len(m.f):
                    x = m(detect_inputs)
                else:
                    # Fallback: use absolute indices from original YAML
                    # Original Detect references [17, 20, 23] which are absolute indices
                    # Map to current head positions
                    original_detect_indices = [17, 20, 23]  # From YAML
                    detect_inputs = []
                    for orig_idx in original_detect_indices:
                        # Map absolute index to current head position
                        head_pos = orig_idx - 10  # Head starts at index 10
                        abs_idx = backbone_len + head_pos
                        if abs_idx in outputs:
                            detect_inputs.append(outputs[abs_idx])
                    
                    if len(detect_inputs) == len(original_detect_indices):
                        x = m(detect_inputs)
                    else:
                        LOGGER.error(f"Detect layer: could not find inputs, found {len(detect_inputs)}/{len(original_detect_indices)}")
                        x = m([x] * len(original_detect_indices))  # Fallback
            else:
                # Regular module - get input from referenced layers
                if m.f != -1:  # if not from previous layer
                    if isinstance(m.f, int):
                        # Single input
                        if m.f == -1:
                            pass  # Use x (previous layer)
                        elif m.f < 0:
                            # Negative index: relative to current head position
                            rel_idx = head_idx + m.f  # m.f is negative, so this subtracts
                            abs_idx = backbone_len + rel_idx
                            if abs_idx in outputs:
                                x = outputs[abs_idx]
                        elif m.f in self.ch_idx_to_fusion:
                            # Converted fusion point index - map back to absolute
                            abs_fusion_idx = self.ch_idx_to_fusion[m.f]
                            if abs_fusion_idx in outputs:
                                x = outputs[abs_fusion_idx]
                        elif m.f in outputs:
                            x = outputs[m.f]
                    else:
                        # Multiple inputs (Concat) - m.f is list of indices
                        concat_inputs = []
                        for j in m.f:
                            if j == -1:
                                concat_inputs.append(x)  # Previous layer
                            elif j < 0:
                                # Negative index: relative to current head position
                                # Convert to absolute index: current_head_idx + j
                                rel_idx = head_idx + j  # j is negative, so this subtracts
                                abs_idx = backbone_len + rel_idx
                                if abs_idx in outputs:
                                    concat_inputs.append(outputs[abs_idx])
                            elif j in self.ch_idx_to_fusion:
                                # This is a converted fusion point index (0, 1, or 2)
                                # Map back to absolute fusion index
                                abs_fusion_idx = self.ch_idx_to_fusion[j]
                                if abs_fusion_idx in outputs:
                                    concat_inputs.append(outputs[abs_fusion_idx])
                            elif j < backbone_len:
                                # Positive index < backbone_len: absolute backbone index
                                # Check if it's a fusion point
                                if j in self.fusion_indices:
                                    # Get the fused feature at this index
                                    fusion_idx = self.fusion_indices.index(j)
                                    if fusion_idx < len(fused_features):
                                        concat_inputs.append(fused_features[fusion_idx])
                                elif j in outputs:
                                    concat_inputs.append(outputs[j])
                            elif j in outputs:
                                # Already processed head layer
                                concat_inputs.append(outputs[j])
                        
                        if len(concat_inputs) > 1:
                            # Ensure all tensors have matching spatial dimensions
                            # Get target spatial size from first tensor
                            target_h, target_w = concat_inputs[0].shape[-2:]
                            for idx, tensor in enumerate(concat_inputs[1:], 1):
                                h, w = tensor.shape[-2:]
                                if h != target_h or w != target_w:
                                    # Interpolate to match spatial dimensions
                                    concat_inputs[idx] = torch.nn.functional.interpolate(
                                        tensor, size=(target_h, target_w), mode='nearest', align_corners=None
                                    )
                            # Concat module expects list/tuple of tensors
                            x = concat_inputs
                        elif len(concat_inputs) == 1:
                            x = concat_inputs[0]
                        else:
                            # No inputs found - use previous layer
                            pass
                
                # Forward through module
                # Concat module's forward expects a list of tensors as single argument
                from models.common import Concat
                if isinstance(m, Concat) and isinstance(x, list):
                    x = m(x)  # Pass list directly to Concat
                else:
                    x = m(x)
            
            # Store output at absolute index
            outputs[abs_idx] = x
        
        return x
    
    def _forward_augment(self, x_rgb, x_ir):
        """Augmented inference (simplified - same transforms for both modalities)."""
        return self._forward_once(x_rgb, x_ir)


# Alias for compatibility
FusionModel = YOLOv5FusionModel

