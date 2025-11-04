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


class LoadFusionImages:
    """
    YOLOv5 Fusion Model image loader for paired RGB and IR images.
    Loads RGB images and finds corresponding IR images by replacing 'visible' with 'infrared'.
    """
    
    def __init__(self, path, img_size=640, stride=32, auto=True):
        """
        Initialize fusion image loader.
        
        Args:
            path: Path to RGB images (directory, file, or glob pattern)
            img_size: Target image size
            stride: Model stride
            auto: Auto-resize
        """
        if isinstance(path, str) and Path(path).suffix == ".txt":
            path = Path(path).read_text().rsplit()
        
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if "*" in p:
                files.extend(sorted(glob.glob(p, recursive=True)))
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, "*.*"))))
            elif os.path.isfile(p):
                files.append(p)
            else:
                raise FileNotFoundError(f"{p} does not exist")
        
        # Filter to only RGB (visible) images that have IR pairs
        self.rgb_files = []
        self.ir_files = []
        for f in files:
            if f.split(".")[-1].lower() in IMG_FORMATS:
                if "visible" in f:
                    ir_file = f.replace("visible", "infrared")
                    if os.path.exists(ir_file):
                        self.rgb_files.append(f)
                        self.ir_files.append(ir_file)
                    else:
                        LOGGER.warning(f"No IR pair found for {f}, skipping")
        
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
        
        # Resize both images
        rgb_im = letterbox(rgb_im0, self.img_size, stride=self.stride, auto=self.auto)[0]
        ir_im = letterbox(ir_im0, self.img_size, stride=self.stride, auto=self.auto)[0]
        
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
        Initialize dual-modal dataset loader.
        
        Args:
            All args same as LoadImagesAndLabels
        """
        # Initialize parent class first
        super().__init__(
            path=path,
            img_size=img_size,
            batch_size=batch_size,
            augment=augment,
            hyp=hyp,
            rect=rect,
            image_weights=image_weights,
            cache_images=cache_images,
            single_cls=single_cls,
            stride=stride,
            pad=pad,
            min_items=min_items,
            prefix=prefix,
            rank=rank,
            seed=seed,
        )
        
        # Store original file lists before filtering (parent class has already set these)
        original_im_files = list(self.im_files)
        original_label_files = list(self.label_files)
        original_labels = list(self.labels)
        original_shapes = self.shapes.copy()
        
        # Filter to only keep visible (RGB) images that have IR pairs
        self.rgb_files = []
        self.ir_files = []
        filtered_indices = []
        
        for i, img_file in enumerate(original_im_files):
            if 'visible' in img_file:
                # This is an RGB image
                ir_file = img_file.replace('visible', 'infrared')
                if os.path.exists(ir_file):
                    self.rgb_files.append(img_file)
                    self.ir_files.append(ir_file)
                    filtered_indices.append(i)
                else:
                    if rank in {-1, 0}:
                        LOGGER.warning(f"{prefix}No IR pair found for {img_file}")
        
        # Update the file lists to filtered pairs
        self.im_files = self.rgb_files
        self.n = len(self.rgb_files)
        
        # Update label files: labels may be stored with _visible or _infrared suffix
        # Convert RGB image paths to label paths by replacing /images/ with /labels/
        # Prefer _infrared label files, fallback to _visible
        from utils.dataloaders import img2label_paths
        label_files_raw = img2label_paths(self.rgb_files)
        self.label_files = []
        for i, (rgb_file, label_file_raw) in enumerate(zip(self.rgb_files, label_files_raw)):
            # First try _infrared label file (preferred)
            label_file_infrared = label_file_raw.replace('_visible', '_infrared')
            if os.path.exists(label_file_infrared):
                label_file = label_file_infrared
            elif os.path.exists(label_file_raw):
                # Fallback to _visible label file
                label_file = label_file_raw
            else:
                # If neither exists, use original (will be empty)
                label_file = label_file_raw
            self.label_files.append(label_file)
        
        # Reload labels with corrected label file paths
        # We need to re-run cache_labels or manually load labels
        # For now, filter labels from original, but we'll need to reload them
        filtered_labels = []
        filtered_shapes = []
        labels_found = 0
        labels_total = 0
        for i, idx in enumerate(filtered_indices):
            # Reload label for this specific file
            label_file = self.label_files[i]
            if os.path.exists(label_file):
                try:
                    with open(label_file) as f:
                        lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                        if len(lb):
                            lb = np.array(lb, dtype=np.float32)
                            if lb.shape[1] == 5:  # class, x, y, w, h
                                filtered_labels.append(lb)
                                labels_found += len(lb)
                                labels_total += len(lb)
                            else:
                                filtered_labels.append(np.zeros((0, 5), dtype=np.float32))
                        else:
                            filtered_labels.append(np.zeros((0, 5), dtype=np.float32))
                except Exception as e:
                    if rank in {-1, 0}:
                        LOGGER.warning(f"{prefix}Error loading label {label_file}: {e}")
                    filtered_labels.append(np.zeros((0, 5), dtype=np.float32))
            else:
                filtered_labels.append(np.zeros((0, 5), dtype=np.float32))
            
            # Use shape from original if available
            if idx < len(original_shapes):
                filtered_shapes.append(original_shapes[idx])
            else:
                filtered_shapes.append((640, 640))  # Default shape
        
        self.labels = filtered_labels
        self.shapes = np.array(filtered_shapes)
        
        # Log label statistics
        if rank in {-1, 0}:
            non_empty = sum(1 for lb in filtered_labels if len(lb) > 0)
            LOGGER.info(f"{prefix}Loaded {len(filtered_labels)} labels ({non_empty} non-empty, {labels_total} total annotations)")
        
        # Recalculate batch indices after filtering
        n = len(self.shapes)
        bi = np.floor(np.arange(n) / batch_size).astype(int)
        nb = bi[-1] + 1
        self.batch = bi
        self.n = n
        
        # CRITICAL: Recreate indices for filtered dataset
        # The parent class created self.indices pointing to original dataset indices
        # After filtering, we need to create new indices pointing to filtered dataset
        # Create mapping from original index to filtered index position
        original_to_filtered = {orig_idx: filt_idx for filt_idx, orig_idx in enumerate(filtered_indices)}
        
        # Filter self.indices to only include indices that exist in filtered dataset
        # and map them to their position in the filtered dataset
        filtered_indices_set = set(filtered_indices)
        new_indices = []
        for orig_idx in self.indices:
            if orig_idx in filtered_indices_set:
                # Map original index to filtered position
                new_indices.append(original_to_filtered[orig_idx])
        
        # Update indices - use mapped indices if we have them, otherwise create fresh ones
        if len(new_indices) > 0:
            self.indices = np.array(new_indices)
        else:
            # Fallback: create fresh indices (shouldn't happen if filtering worked)
            self.indices = np.arange(n)
        
        # Handle DDP case - need to re-filter for DDP after mapping
        if rank > -1:  # DDP mode
            from utils.dataloaders import WORLD_SIZE
            # Filter indices to match DDP rank
            self.indices = self.indices[np.random.RandomState(seed=seed).permutation(len(self.indices)) % WORLD_SIZE == RANK]
        
        # Also need to update batch_shapes if rect mode is used
        if self.rect:
            # Recalculate batch shapes for filtered dataset
            # Similar to parent class logic
            s = self.shapes  # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.shapes = s[irect]  # wh
            self.im_files = [self.im_files[i] for i in irect]
            self.rgb_files = [self.rgb_files[i] for i in irect]
            self.ir_files = [self.ir_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            if hasattr(self, 'segments'):
                self.segments = [self.segments[i] for i in irect]
            ar = ar[irect]
            # Set training image shapes
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
        
        # Store cache setting
        self.cache_images_setting = cache_images
        
        # Reinitialize imgs cache to match filtered files
        if cache_images:
            self.imgs = [None] * self.n
            self.ir_imgs = [None] * self.n  # Separate cache for IR images
        else:
            self.imgs = [None] * self.n
            self.ir_imgs = None
        
        # NOTE: self.indices was already set above with mapped indices - DO NOT overwrite!
        
        if rank in {-1, 0}:
            LOGGER.info(f"{prefix}Found {self.n} RGB-IR image pairs (cache: {cache_images})")
    
    def load_image_pair(self, index):
        """
        Load paired RGB and IR images.
        
        Args:
            index: Dataset index
            
        Returns:
            rgb_img: RGB image (H, W, 3)
            ir_img: IR image (H, W, 3)
            (h0, w0): Original image size
            (h, w): Resized image size
        """
        # Load RGB image
        rgb_path = self.rgb_files[index]
        rgb_img = self.imgs[index] if self.cache_images_setting and self.imgs[index] is not None else None
        
        if rgb_img is None:  # not cached
            rgb_img = cv2.imread(rgb_path)  # BGR
            if rgb_img is None:
                raise FileNotFoundError(f"Image not found: {rgb_path}")
            h0, w0 = rgb_img.shape[:2]  # orig hw
            
            # Resize
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                rgb_img = cv2.resize(rgb_img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            
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
            
            # Resize to same size as RGB (same ratio)
            r = self.img_size / max(h0, w0)
            if r != 1:
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                ir_img = cv2.resize(ir_img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            
            # Cache if enabled
            if self.cache_images_setting:
                if self.ir_imgs is None:
                    self.ir_imgs = [None] * self.n
                self.ir_imgs[index] = ir_img
        
        h, w = rgb_img.shape[:2]
        return rgb_img, ir_img, (h0, w0), (h, w)
    
    def __getitem__(self, index):
        """Fetches paired RGB-IR dataset item with synchronized augmentation."""
        # Map dataset index to filtered dataset index
        # self.indices now contains filtered indices directly
        mapped_index = self.indices[index] if hasattr(self, 'indices') and len(self.indices) > index else index
        
        hyp = self.hyp
        mosaic = self.mosaic and random.random() < hyp["mosaic"]
        
        if mosaic:
            # Mosaic augmentation - use parent's load_mosaic but adapt for dual inputs
            # For simplicity, disable mosaic for now (can be enhanced later)
            # Load single pair instead
            rgb_img, ir_img, (h0, w0), (h, w) = self.load_image_pair(mapped_index)
            labels = self.labels[mapped_index].copy()
            
            # In mosaic path, labels are in normalized xywh format
            # Need to convert to xyxy format for random_perspective
            # Also need to letterbox the image
            shape = self.img_size
            rgb_img, ratio, pad = letterbox(rgb_img, shape, auto=False, scaleup=self.augment)
            ir_img, _, _ = letterbox(ir_img, shape, auto=False, scaleup=self.augment)
            
            # Convert labels from normalized xywh to pixel xyxy
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
            
            shapes = None
        else:
            # Load single pair
            rgb_img, ir_img, (h0, w0), (h, w) = self.load_image_pair(mapped_index)
            
            # Letterbox
            shape = self.batch_shapes[self.batch[mapped_index]] if self.rect and self.batch_shapes is not None else self.img_size
            rgb_img, ratio, pad = letterbox(rgb_img, shape, auto=False, scaleup=self.augment)
            ir_img, _, _ = letterbox(ir_img, shape, auto=False, scaleup=self.augment)  # Same shape
            
            shapes = (h0, w0), ((h / h0, w / w0), pad)
            
            labels = self.labels[mapped_index].copy()
            
            if labels.size:  # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])
        
        # Synchronized augmentation
        if self.augment:
            # Random perspective (same transform for both)
            # Store random state before first transform
            random_state = random.getstate()
            rgb_img, labels = random_perspective(
                rgb_img,
                labels,
                degrees=hyp["degrees"],
                translate=hyp["translate"],
                scale=hyp["scale"],
                shear=hyp["shear"],
                perspective=hyp["perspective"],
            )
            # Apply same transform to IR by restoring random state
            # Note: This ensures same random parameters but not exact same matrix
            # For perfect synchronization, we'd need to capture and reuse the transform matrix
            random.setstate(random_state)
            ir_img, _ = random_perspective(
                ir_img,
                labels.copy(),  # Dummy labels for IR (not used in transform)
                degrees=hyp["degrees"],
                translate=hyp["translate"],
                scale=hyp["scale"],
                shear=hyp["shear"],
                perspective=hyp["perspective"],
            )
            
            # HSV augmentation (RGB only, IR doesn't have color)
            augment_hsv(rgb_img, hgain=hyp["hsv_h"], sgain=hyp["hsv_s"], vgain=hyp["hsv_v"])
            
            # Synchronized flips
            if random.random() < hyp["flipud"]:
                rgb_img = np.flipud(rgb_img)
                ir_img = np.flipud(ir_img)
                if len(labels):
                    labels[:, 2] = 1 - labels[:, 2]
            
            if random.random() < hyp["fliplr"]:
                rgb_img = np.fliplr(rgb_img)
                ir_img = np.fliplr(ir_img)
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]
        
        nl = len(labels)  # number of labels
        if nl:
            labels[:, 1:5] = xyxy2xywhn(labels[:, 1:5], w=rgb_img.shape[1], h=rgb_img.shape[0], clip=True, eps=1e-3)
        
        labels_out = torch.zeros((nl, 6))
        if nl:
            labels_out[:, 1:] = torch.from_numpy(labels)
        else:
            # Empty labels - create empty tensor with correct shape
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

