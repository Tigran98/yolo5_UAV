"""
Dataloader for YOLOv5 Fusion Model - Handles paired RGB and IR images
"""

import glob
import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import math
from torch.utils.data import DataLoader, distributed

# Import from existing dataloader
from utils.dataloaders import (
    IMG_FORMATS,
    LoadImagesAndLabels,
    letterbox,
    xywhn2xyxy,
    xyxy2xywhn,
    random_perspective,
    augment_hsv,
    seed_worker,
    SmartDistributedSampler,
    InfiniteDataLoader,
    PIN_MEMORY,
)
from utils.general import LOGGER
from utils.torch_utils import torch_distributed_zero_first

RANK = int(os.getenv("RANK", -1))


def apply_synchronized_perspective(rgb_img, ir_img, labels, degrees=10, translate=0.1, scale=0.1, 
                                   shear=10, perspective=0.0, border=(0, 0)):
    """
    Apply synchronized perspective transformation to both RGB and IR images.
    Generates transformation matrix once and applies it to both images.
    
    Args:
        rgb_img: RGB image (H, W, 3)
        ir_img: IR image (H, W, 3)
        labels: Labels array (n, 5) in format [class, x1, y1, x2, y2] (pixel coordinates)
        degrees, translate, scale, shear, perspective: Augmentation parameters
        border: Border padding
        
    Returns:
        rgb_img_transformed: Transformed RGB image
        ir_img_transformed: Transformed IR image
        labels_transformed: Transformed labels
    """
    height = rgb_img.shape[0] + border[0] * 2
    width = rgb_img.shape[1] + border[1] * 2
    
    # Generate transformation matrix (same for both RGB and IR)
    # Center
    C = np.eye(3)
    C[0, 2] = -rgb_img.shape[1] / 2
    C[1, 2] = -rgb_img.shape[0] / 2
    
    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)
    
    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
    
    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)
    
    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height
    
    # Combined transformation matrix
    M = T @ S @ R @ P @ C
    
    # Apply same transformation to RGB
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            rgb_img_transformed = cv2.warpPerspective(rgb_img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            rgb_img_transformed = cv2.warpAffine(rgb_img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
    else:
        rgb_img_transformed = rgb_img.copy()
    
    # Apply same transformation to IR
    if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():
        if perspective:
            ir_img_transformed = cv2.warpPerspective(ir_img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            ir_img_transformed = cv2.warpAffine(ir_img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))
    else:
        ir_img_transformed = ir_img.copy()
    
    # Transform labels
    labels_transformed = labels.copy()
    if len(labels):
        n = len(labels)
        xy = np.ones((n * 4, 3))
        xy[:, :2] = labels[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T
        xy = (xy[:, :2] / xy[:, 2:3] if perspective else xy[:, :2]).reshape(n, 8)
        
        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        
        # Clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)
        
        # Filter candidates (from original random_perspective logic)
        from utils.augmentations import box_candidates
        # Use scale factor s from rotation matrix calculation above
        i = box_candidates(box1=labels[:, 1:5].T * s, box2=new.T, area_thr=0.10)
        if len(i) > 0:
            labels_transformed = labels_transformed[i]
            labels_transformed[:, 1:5] = new[i]
        else:
            labels_transformed = np.zeros((0, 5), dtype=np.float32)
    
    return rgb_img_transformed, ir_img_transformed, labels_transformed


class LoadFusionImages:
    """
    YOLOv5 Fusion Model image loader for paired RGB and IR images.
    Loads RGB and IR images from separate rgb/ and ir/ directories.
    Images are already 640x640, no resizing needed.
    """
    
    def __init__(self, path, img_size=640, stride=32, auto=True):
        """
        Initialize fusion image loader.
        
        Args:
            path: Path to images directory (should contain rgb/ and ir/ subdirectories)
            img_size: Target image size (640, images already resized)
            stride: Model stride
            auto: Auto-resize (ignored, images already 640x640)
        """
        path = Path(path)
        
        # Find RGB and IR directories
        rgb_dir = path / 'rgb'
        ir_dir = path / 'ir'
        
        if not rgb_dir.exists():
            raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")
        if not ir_dir.exists():
            raise FileNotFoundError(f"IR directory not found: {ir_dir}")
        
        # Discover RGB images
        rgb_files = sorted(glob.glob(str(rgb_dir / "*.*")))
        rgb_files = [f for f in rgb_files if f.split(".")[-1].lower() in IMG_FORMATS]
        
        # Discover IR images and match by filename
        ir_files_dict = {}
        ir_files_list = sorted(glob.glob(str(ir_dir / "*.*")))
        for ir_file in ir_files_list:
            if ir_file.split(".")[-1].lower() in IMG_FORMATS:
                ir_name = Path(ir_file).stem
                ir_files_dict[ir_name] = ir_file
        
        # Match RGB and IR files by base filename
        self.rgb_files = []
        self.ir_files = []
        for rgb_file in rgb_files:
            rgb_name = Path(rgb_file).stem
            if rgb_name in ir_files_dict:
                self.rgb_files.append(rgb_file)
                self.ir_files.append(ir_files_dict[rgb_name])
            else:
                LOGGER.warning(f"No IR pair found for RGB image: {rgb_file}, skipping")
        
        self.nf = len(self.rgb_files)
        assert self.nf > 0, f"No RGB-IR image pairs found in {path}"
        
        self.img_size = img_size
        self.stride = stride
        self.auto = auto
        self.mode = "image"
    
    def __iter__(self):
        """Initialize iterator."""
        self.count = 0
        return self
    
    def __next__(self):
        """Load next RGB-IR image pair."""
        if self.count == self.nf:
            raise StopIteration
        
        rgb_path = self.rgb_files[self.count]
        ir_path = self.ir_files[self.count]
        self.count += 1
        
        # Load RGB image
        rgb_im0 = cv2.imread(rgb_path)  # BGR
        assert rgb_im0 is not None, f"Image Not Found {rgb_path}"
        
        # Load IR image
        ir_im0 = cv2.imread(ir_path)  # BGR
        assert ir_im0 is not None, f"IR Image Not Found {ir_path}"
        
        # Images are already 640x640, no letterbox/resizing needed
        rgb_im = rgb_im0.copy()
        ir_im = ir_im0.copy()
        
        # Convert to CHW, BGR to RGB
        rgb_im = rgb_im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        rgb_im = np.ascontiguousarray(rgb_im)
        
        ir_im = ir_im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        ir_im = np.ascontiguousarray(ir_im)
        
        s = f"image {self.count}/{self.nf} {rgb_path}: "
        
        return rgb_path, rgb_im, ir_im, rgb_im0, ir_im0, s
    
    def __len__(self):
        """Return number of image pairs."""
        return self.nf


class LoadFusionVideo:
    """
    YOLOv5 Fusion Model video loader for RGB and IR videos.
    Supports:
    - Single RGB video (IR simulated from RGB grayscale)
    - Dual RGB+IR videos (proper RGB-to-IR alignment preprocessing)
    """
    
    def __init__(self, rgb_path, ir_path=None, img_size=640, stride=32, auto=True, vid_stride=1):
        """
        Initialize fusion video loader.
        
        Args:
            rgb_path: Path to RGB video file
            ir_path: Optional path to IR video file. If None, IR is simulated from RGB (grayscale)
            img_size: Target image size (640)
            stride: Model stride
            auto: Auto-resize
            vid_stride: Video frame stride (process every nth frame)
        """
        rgb_path = str(Path(rgb_path).resolve())
        
        self.img_size = img_size
        self.stride = stride
        self.auto = auto
        self.vid_stride = vid_stride
        self.rgb_path = rgb_path
        self.ir_path = str(Path(ir_path).resolve()) if ir_path else None
        self.mode = "video"
        self.has_ir_video = ir_path is not None
        
        # Open RGB video
        self.rgb_cap = cv2.VideoCapture(rgb_path)
        assert self.rgb_cap.isOpened(), f"Failed to open RGB video: {rgb_path}"
        
        # Open IR video if provided
        if self.has_ir_video:
            self.ir_cap = cv2.VideoCapture(self.ir_path)
            assert self.ir_cap.isOpened(), f"Failed to open IR video: {self.ir_path}"
        else:
            self.ir_cap = None
        
        # Get video properties from RGB video
        self.frames = int(self.rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.rgb_cap.get(cv2.CAP_PROP_FPS)
        self.w = int(self.rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Verify IR video matches RGB video properties if provided
        if self.has_ir_video:
            ir_frames = int(self.ir_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ir_fps = self.ir_cap.get(cv2.CAP_PROP_FPS)
            ir_w = int(self.ir_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ir_h = int(self.ir_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if ir_frames != self.frames:
                LOGGER.warning(f"IR video has {ir_frames} frames, RGB has {self.frames} frames. Using minimum.")
                self.frames = min(ir_frames, self.frames)
            
            LOGGER.info(f"RGB Video: {self.frames} frames ({self.w}x{self.h}) at {self.fps:.2f} FPS")
            LOGGER.info(f"IR Video: {ir_frames} frames ({ir_w}x{ir_h}) at {ir_fps:.2f} FPS")
        else:
            LOGGER.info(f"Video: {self.frames} frames ({self.w}x{self.h}) at {self.fps:.2f} FPS (IR simulated from RGB)")
        
        self.count = 0
        self.frame = 0
    
    def __iter__(self):
        """Initialize iterator."""
        self.count = 0
        self.frame = 0
        self.rgb_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        if self.has_ir_video:
            self.ir_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning
        return self
    
    def __next__(self):
        """Load next video frame."""
        # Apply vid_stride to RGB video
        for _ in range(self.vid_stride):
            ret_val = self.rgb_cap.grab()
            if not ret_val:
                raise StopIteration
        
        ret_val, rgb_im0 = self.rgb_cap.retrieve()
        if not ret_val:
            raise StopIteration
        
        # Load IR frame if IR video is provided
        if self.has_ir_video:
            # Apply same vid_stride to IR video
            for _ in range(self.vid_stride):
                ret_val_ir = self.ir_cap.grab()
                if not ret_val_ir:
                    raise StopIteration
            
            ret_val_ir, ir_im0 = self.ir_cap.retrieve()
            if not ret_val_ir:
                raise StopIteration
        else:
            # Convert RGB frame to grayscale for IR simulation (3-channel grayscale)
            ir_im0 = cv2.cvtColor(rgb_im0, cv2.COLOR_BGR2GRAY)
            ir_im0 = cv2.cvtColor(ir_im0, cv2.COLOR_GRAY2BGR)  # Convert back to 3-channel
        
        self.frame += self.vid_stride
        
        # Apply preprocessing matching training pipeline
        # Ensure both RGB and IR are exactly the same size (640x640) for fusion
        from utils.dataloaders import letterbox
        
        # Resize both RGB and IR to exactly img_size x img_size
        # Use consistent preprocessing to ensure spatial dimensions match exactly
        # This is critical for fusion - both must be exactly the same size
        # Use auto=False to ensure exact size without stride rounding differences
        # Handle img_size as int or list
        img_size_val = self.img_size[0] if isinstance(self.img_size, (list, tuple)) else self.img_size
        rgb_im, rgb_ratio, rgb_pad = letterbox(rgb_im0, img_size_val, stride=self.stride, auto=False)
        ir_im, ir_ratio, ir_pad = letterbox(ir_im0, img_size_val, stride=self.stride, auto=False)
        
        # Verify both are exactly the same size (should be guaranteed with auto=False)
        target_size = (img_size_val, img_size_val)
        assert rgb_im.shape[:2] == ir_im.shape[:2] == target_size, \
            f"Size mismatch: RGB={rgb_im.shape[:2]}, IR={ir_im.shape[:2]}, target={target_size}"
        
        # Store preprocessing info for coordinate transformation
        # ratio = (r, r) where r = new/old, pad = (dw, dh)
        # For scale_boxes: gain = old/new = 1/r, pad = (dw, dh)
        self.rgb_ratio_pad = ((1.0 / rgb_ratio[0], 1.0 / rgb_ratio[1]), rgb_pad) if rgb_ratio[0] > 0 else ((1.0, 1.0), (0.0, 0.0))
        self.ir_ratio_pad = ((1.0 / ir_ratio[0], 1.0 / ir_ratio[1]), ir_pad) if ir_ratio[0] > 0 else ((1.0, 1.0), (0.0, 0.0))
        self.rgb_im0_shape = rgb_im0.shape[:2]  # (H, W)
        self.ir_im0_shape = ir_im0.shape[:2]  # (H, W)
        
        # Convert to CHW, BGR to RGB
        rgb_im = rgb_im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        rgb_im = np.ascontiguousarray(rgb_im)
        
        ir_im = ir_im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        ir_im = np.ascontiguousarray(ir_im)
        
        s = f"video frame {self.frame}/{self.frames} {self.rgb_path}: "
        
        return self.rgb_path, rgb_im, ir_im, rgb_im0, ir_im0, s
    
    def __len__(self):
        """Return number of frames."""
        return self.frames // self.vid_stride
    
    def __del__(self):
        """Release video capture."""
        if hasattr(self, 'rgb_cap'):
            self.rgb_cap.release()
        if hasattr(self, 'ir_cap') and self.ir_cap is not None:
            self.ir_cap.release()


class DualModalDataset(LoadImagesAndLabels):
    """
    Dataset loader for paired RGB and IR images for fusion model.
    Extends LoadImagesAndLabels to handle dual modalities.
    """

    def __init__(
        self,
        path,
        img_size=640,
        batch_size=16,
        augment=False,
        hyp=None,
        rect=False,
        image_weights=False,
        cache_images=False,
        single_cls=False,
        stride=32,
        pad=0.0,
        min_items=0,
        prefix="",
        rank=-1,
        seed=0,
    ):
        """
        Initialize dual-modal dataset loader for unified labeling format.
        
        Expected structure:
        - path/images/{set}_resized/rgb/  (RGB images)
        - path/images/{set}_resized/ir/   (IR images)
        - path/labels/{set}_resized/      (unified labels)
        
        Args:
            All args same as LoadImagesAndLabels
        """
        # Set basic attributes before calling parent
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path
        
        # Discover RGB and IR files from separate directories
        # Path format: root/images/{set}_resized
        path = Path(path) if isinstance(path, str) else path
        path_str = str(path.resolve())
        
        # Find RGB and IR directories
        rgb_dir = path / 'rgb'
        ir_dir = path / 'ir'
        
        if not rgb_dir.exists():
            raise FileNotFoundError(f"{prefix}RGB directory not found: {rgb_dir}")
        if not ir_dir.exists():
            raise FileNotFoundError(f"{prefix}IR directory not found: {ir_dir}")
        
        # Discover RGB images
        rgb_files = sorted(glob.glob(str(rgb_dir / "*.*")))
        rgb_files = [f for f in rgb_files if f.split(".")[-1].lower() in IMG_FORMATS]
        
        # Discover IR images and match by filename
        ir_files_dict = {}
        ir_files_list = sorted(glob.glob(str(ir_dir / "*.*")))
        for ir_file in ir_files_list:
            if ir_file.split(".")[-1].lower() in IMG_FORMATS:
                ir_name = Path(ir_file).stem  # filename without extension
                ir_files_dict[ir_name] = ir_file
        
        # Match RGB and IR files by base filename
        self.rgb_files = []
        self.ir_files = []
        matched_names = []
        
        for rgb_file in rgb_files:
            rgb_name = Path(rgb_file).stem
            if rgb_name in ir_files_dict:
                self.rgb_files.append(rgb_file)
                self.ir_files.append(ir_files_dict[rgb_name])
                matched_names.append(rgb_name)
            else:
                if rank in {-1, 0}:
                    LOGGER.warning(f"{prefix}No IR pair found for RGB image: {rgb_file}")
        
        assert len(self.rgb_files) > 0, f"{prefix}No RGB-IR pairs found in {path}"
        
        # Find label directory (unified labels)
        # Extract set name from path: images/{set}_resized -> labels/{set}_resized
        path_parts = path_str.split(os.sep)
        set_name = None
        for part in reversed(path_parts):
            if part.endswith('_resized'):
                set_name = part
                break
        
        if set_name is None:
            # Fallback: try to find labels directory relative to images
            label_dir = path.parent.parent / 'labels' / path.name
        else:
            # Construct label path: replace images/{set}_resized with labels/{set}_resized
            label_dir = Path(path_str.replace(os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep))
        
        if not label_dir.exists():
            raise FileNotFoundError(f"{prefix}Label directory not found: {label_dir}")
        
        # Load unified labels
        self.label_files = []
        self.labels = []
        self.shapes = []
        labels_found = 0
        labels_total = 0
        
        for rgb_name in matched_names:
            label_file = label_dir / f"{rgb_name}.txt"
            self.label_files.append(str(label_file))
            
            if label_file.exists():
                try:
                    with open(label_file) as f:
                        lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        if len(lb):
                            lb = np.array(lb, dtype=np.float32)
                            if lb.shape[1] == 5:  # class, x, y, w, h
                                self.labels.append(lb)
                                labels_found += len(lb)
                                labels_total += len(lb)
                            else:
                                self.labels.append(np.zeros((0, 5), dtype=np.float32))
                        else:
                            self.labels.append(np.zeros((0, 5), dtype=np.float32))
                except Exception as e:
                    if rank in {-1, 0}:
                        LOGGER.warning(f"{prefix}Error loading label {label_file}: {e}")
                    self.labels.append(np.zeros((0, 5), dtype=np.float32))
            else:
                self.labels.append(np.zeros((0, 5), dtype=np.float32))
            
            # Images are already 640x640, no resizing needed
            self.shapes.append((640, 640))
        
        self.shapes = np.array(self.shapes)
        self.im_files = self.rgb_files  # For compatibility with parent class
        
        # Log label statistics
        if rank in {-1, 0}:
            non_empty = sum(1 for lb in self.labels if len(lb) > 0)
            LOGGER.info(f"{prefix}Loaded {len(self.labels)} unified labels ({non_empty} non-empty, {labels_total} total annotations)")
        
        # Create indices
        n = len(self.shapes)
        bi = np.floor(np.arange(n) / batch_size).astype(int)
        nb = bi[-1] + 1
        self.batch = bi
        self.n = n
        self.indices = np.arange(n)
        
        # Handle DDP case
        if rank > -1:
            from utils.dataloaders import WORLD_SIZE
            self.indices = self.indices[np.random.RandomState(seed=seed).permutation(n) % WORLD_SIZE == RANK]
        
        # Rectangular Training
        if self.rect:
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.shapes = s[irect]
            self.im_files = [self.im_files[i] for i in irect]
            self.rgb_files = [self.rgb_files[i] for i in irect]
            self.ir_files = [self.ir_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            if hasattr(self, 'segments'):
                self.segments = [self.segments[i] for i in irect]
            ar = ar[irect]
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                if len(ari) > 0:
                    mini, maxi = ari.min(), ari.max()
                    if maxi < 1:
                        shapes[i] = [maxi, 1]
                    elif mini > 1:
                        shapes[i] = [1, 1 / mini]
            self.batch_shapes = np.ceil(np.array(shapes) * img_size / stride + pad).astype(int) * stride
        else:
            self.batch_shapes = None
        
        # Single class handling
        if single_cls:
            for i in range(len(self.labels)):
                if len(self.labels[i]) > 0:
                    self.labels[i][:, 0] = 0
        
        # Cache images
        self.cache_images_setting = cache_images
        if cache_images:
            self.imgs = [None] * self.n
            self.ir_imgs = [None] * self.n
        else:
            self.imgs = [None] * self.n
            self.ir_imgs = None
        
        # Initialize segments (for compatibility)
        self.segments = [None] * self.n
        
        if rank in {-1, 0}:
            LOGGER.info(f"{prefix}Found {self.n} RGB-IR image pairs (cache: {cache_images})")
    
    def load_mosaic_pair(self, index):
        """
        Load synchronized 4-image mosaic for both RGB and IR.
        Uses same indices, same mosaic center, and same layout for both modalities.
        
        Args:
            index: Primary image index
            
        Returns:
            rgb_img4: RGB mosaic image (1280x1280)
            ir_img4: IR mosaic image (1280x1280)
            labels4: Combined labels from 4 images
        """
        labels4 = []
        s = self.img_size  # 640
        # Same mosaic center for both RGB and IR
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)
        
        # Select same 4 indices for both RGB and IR
        indices = [index] + random.choices(self.indices, k=3)
        random.shuffle(indices)
        
        # Initialize mosaic images
        rgb_img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        ir_img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)
        
        for i, idx in enumerate(indices):
            # Load RGB-IR pair
            rgb_img, ir_img, (h0, w0), (h, w) = self.load_image_pair(idx)
            
            # Same layout coordinates for both RGB and IR
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            # Place images in mosaic (same coordinates for both)
            rgb_img4[y1a:y2a, x1a:x2a] = rgb_img[y1b:y2b, x1b:x2b]
            ir_img4[y1a:y2a, x1a:x2a] = ir_img[y1b:y2b, x1b:x2b]
            
            padw = x1a - x1b
            padh = y1a - y1b
            
            # Transform labels
            labels = self.labels[idx].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w=640, h=640, padw=padw, padh=padh)
            labels4.append(labels)
        
        # Concatenate and clip labels
        labels4 = np.concatenate(labels4, 0)
        np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])
        
        return rgb_img4, ir_img4, labels4
    
    def load_image_pair(self, index):
        """
        Load paired RGB and IR images.
        Images are already 640x640, no resizing needed.
        
        Args:
            index: Dataset index
            
        Returns:
            rgb_img: RGB image (H, W, 3) - already 640x640
            ir_img: IR image (H, W, 3) - already 640x640
            (h0, w0): Original image size (640, 640)
            (h, w): Resized image size (640, 640)
        """
        # Load RGB image
        rgb_path = self.rgb_files[index]
        rgb_img = self.imgs[index] if self.cache_images_setting and self.imgs[index] is not None else None
        
        if rgb_img is None:  # not cached
            rgb_img = cv2.imread(rgb_path)  # BGR
            if rgb_img is None:
                raise FileNotFoundError(f"Image not found: {rgb_path}")
            
            # Verify image is 640x640 (should already be resized)
            h0, w0 = rgb_img.shape[:2]
            if h0 != 640 or w0 != 640:
                LOGGER.warning(f"RGB image {rgb_path} is {w0}x{h0}, expected 640x640")
            
            # Cache if enabled
            if self.cache_images_setting:
                self.imgs[index] = rgb_img
        else:
            h0, w0 = rgb_img.shape[:2]
        
        # Load IR image
        ir_path = self.ir_files[index]
        ir_img = None
        if self.cache_images_setting and self.ir_imgs is not None:
            ir_img = self.ir_imgs[index]
        
        if ir_img is None:  # not cached
            ir_img = cv2.imread(ir_path)  # BGR
            if ir_img is None:
                raise FileNotFoundError(f"IR image not found: {ir_path}")
            
            # Verify image is 640x640 (should already be resized)
            ir_h, ir_w = ir_img.shape[:2]
            if ir_h != 640 or ir_w != 640:
                LOGGER.warning(f"IR image {ir_path} is {ir_w}x{ir_h}, expected 640x640")
            
            # Cache if enabled
            if self.cache_images_setting:
                if self.ir_imgs is None:
                    self.ir_imgs = [None] * self.n
                self.ir_imgs[index] = ir_img
        
        h, w = rgb_img.shape[:2]
        return rgb_img, ir_img, (h0, w0), (h, w)
    
    def __getitem__(self, index):
        """Fetches paired RGB-IR dataset item with synchronized augmentation.
        Images are already 640x640, no letterbox/resizing needed.
        """
        mapped_index = self.indices[index] if hasattr(self, 'indices') and len(self.indices) > index else index
        
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        
        if mosaic:
            # Synchronized mosaic augmentation - load 4 RGB-IR pairs with same layout
            rgb_img, ir_img, labels = self.load_mosaic_pair(mapped_index)
            shapes = None
        else:
            # Load single pair
            rgb_img, ir_img, (h0, w0), (h, w) = self.load_image_pair(mapped_index)
            
            # Images are already 640x640, no letterbox needed
            shapes = (h0, w0), ((1.0, 1.0), (0, 0))  # No scaling or padding
            
            labels = self.labels[mapped_index].copy()
            
            # Convert labels from normalized xywh to pixel xyxy format
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], w=640, h=640, padw=0, padh=0)
        
        # Synchronized augmentation
        if self.augment:
            if mosaic:
                # For mosaic, use border for perspective transform
                border = self.mosaic_border
            else:
                border = (0, 0)
            
            # Synchronized random perspective (same transform matrix for both RGB and IR)
            rgb_img, ir_img, labels = apply_synchronized_perspective(
                rgb_img, ir_img, labels,
                degrees=hyp["degrees"],
                translate=hyp["translate"],
                scale=hyp["scale"],
                shear=hyp["shear"],
                perspective=hyp["perspective"],
                border=border
            )
            
            # HSV augmentation (RGB only, IR doesn't have color)
            augment_hsv(rgb_img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])
            
            # Synchronized flips (same random decision for both)
            flip_ud = random.random() < hyp["flipud"]
            if flip_ud:
                rgb_img = np.flipud(rgb_img)
                ir_img = np.flipud(ir_img)
                if len(labels):
                    labels[:, 2] = 1 - labels[:, 2]  # Flip y coordinates
            
            flip_lr = random.random() < hyp["fliplr"]
            if flip_lr:
                rgb_img = np.fliplr(rgb_img)
                ir_img = np.fliplr(ir_img)
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]  # Flip x coordinates
        
        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=rgb_img.shape[1], h=rgb_img.shape[0], clip=True, eps=1e-3)
        
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        else:
            labels_out = torch.zeros((0, 6))
        
        # Convert to CHW, BGR to RGB
        rgb_img = rgb_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        rgb_img = np.ascontiguousarray(rgb_img)
        
        ir_img = ir_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        ir_img = np.ascontiguousarray(ir_img)
        
        paths = (self.rgb_files[mapped_index], self.ir_files[mapped_index])
        return torch.from_numpy(rgb_img), torch.from_numpy(ir_img), labels_out, paths, shapes
    
    @staticmethod
    def collate_fn(batch):
        """Collate function for dual-input batches."""
        rgb_imgs, ir_imgs, labels, paths, shapes = zip(*batch)
        # Set image indices for all labels - match parent class behavior exactly
        for i, lb in enumerate(labels):
            if lb.shape[0] > 0:  # Only set image index if labels exist
                lb[:, 0] = i  # add target image index for build_targets()
        # Concatenate labels - match parent class behavior
        # Parent class does: torch.cat(label, 0) which handles empty tensors
        labels_cat = torch.cat(labels, 0)
        return torch.stack(rgb_imgs, 0), torch.stack(ir_imgs, 0), labels_cat, paths, shapes
    
    @staticmethod
    def collate_fn4(batch):
        """Collate function for dual-input batches with 4x mosaic."""
        rgb_imgs, ir_imgs, labels, paths, shapes = zip(*batch)
        n = len(shapes) // 4
        rgb_imgs4, ir_imgs4, labels4, paths4, shapes4 = [], [], [], [], []
        ho = torch.tensor([[0.0, 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0.0, 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, 0.5, 0.5, 0.5, 0.5]])  # scale
        for i in range(n):  # zidane torch.zeros(16,3,720,1280)  # BCHW
            i *= 4
            if random.random() < 0.5:
                im = rgb_imgs[i].numpy()  # HWC
                ir = ir_imgs[i].numpy()
                h, w = im.shape[1:]
                labels4.append(torch.zeros(0, 6))
                s_ = s[0]
                im, ir = im.copy(), ir.copy()
                if random.random() < 0.5:
                    im = cv2.resize(im, (int(w * s_), int(h * s_)), interpolation=cv2.INTER_LINEAR)
                    ir = cv2.resize(ir, (int(w * s_), int(h * s_)), interpolation=cv2.INTER_LINEAR)
                else:
                    im = cv2.resize(im, (int(w / s_), int(h / s_)), interpolation=cv2.INTER_LINEAR)
                    ir = cv2.resize(ir, (int(w / s_), int(h / s_)), interpolation=cv2.INTER_LINEAR)
                top, left = int(random.uniform(0, h - im.shape[1])), int(random.uniform(0, w - im.shape[2]))
                im = im[:, top : top + im.shape[1], left : left + im.shape[2]]
                ir = ir[:, top : top + ir.shape[1], left : left + ir.shape[2]]
                rgb_imgs4.append(torch.from_numpy(im))
                ir_imgs4.append(torch.from_numpy(ir))
            else:
                labels4.append(labels[i])  # no augmentation needed
                rgb_imgs4.append(rgb_imgs[i])
                ir_imgs4.append(ir_imgs[i])
        for i, label in enumerate(labels4):
            label[:, 0] = i  # add target image index for build_targets()
        return torch.stack(rgb_imgs4, 0), torch.stack(ir_imgs4, 0), torch.cat(labels4, 0), paths4, shapes4


def create_fusion_dataloader(
    path,
    imgsz,
    batch_size,
    stride=32,
    single_cls=False,
    hyp=None,
    augment=False,
    cache=False,
    pad=0.0,
    rect=False,
    rank=-1,
    workers=8,
    image_weights=False,
    quad=False,
    prefix="",
    shuffle=False,
    seed=0,
):
    """
    Create dataloader for fusion model (dual RGB-IR inputs).
    
    Args:
        path: Dataset path or path list
        imgsz: Image size
        batch_size: Batch size
        Other args: Same as create_dataloader
        
    Returns:
        DataLoader with dual inputs
    """
    if rect and shuffle:
        LOGGER.warning("WARNING ⚠️ --rect is incompatible with DataLoader shuffle, setting shuffle=False")
        shuffle = False
    with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
        dataset = DualModalDataset(
            path=path,
            img_size=imgsz,
            batch_size=batch_size,
            augment=augment,
            hyp=hyp,
            rect=rect,
            cache_images=cache,
            single_cls=single_cls,
            stride=int(stride),
            pad=pad,
            image_weights=image_weights,
            prefix=prefix,
            rank=rank,
            seed=seed,
        )
    
    batch_size = min(batch_size, len(dataset))
    nd = torch.cuda.device_count()  # number of CUDA devices
    nw = min([os.cpu_count() // max(nd, 1), batch_size if batch_size > 1 else 0, workers])  # number of workers
    sampler = None if rank == -1 else SmartDistributedSampler(dataset, shuffle=shuffle)
    loader = DataLoader if image_weights else InfiniteDataLoader  # only DataLoader allows for attribute updates
    generator = torch.Generator()
    generator.manual_seed(6148914691236517205 + seed + RANK)
    return loader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        num_workers=nw,
        sampler=sampler,
        drop_last=quad,
        pin_memory=PIN_MEMORY,
        collate_fn=DualModalDataset.collate_fn4 if quad else DualModalDataset.collate_fn,
        worker_init_fn=seed_worker,
        generator=generator,
    ), dataset

