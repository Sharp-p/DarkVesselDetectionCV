"""
data.py

Handles dataset loading, SAR preprocessing (Sentinel-1 VV/VH backscatter calibration),
auxiliary context channel integration (bathymetry, distance to shore via Euclidean Distance Transform,
and ocean wind speed from OWI products), SAR-safe data augmentation, and PyTorch DataLoader
construction for Dark Vessel Detection (CGCAM).

Contracts:
    Model Input:  (Batch_Size, 5, 256, 256) float32 in [0.0, 1.0]
        - Channel 0: VH polarization backscatter (dB normalized in [-35.0, +10.0])
        - Channel 1: VV polarization backscatter (dB normalized in [-30.0, +15.0])
        - Channel 2: Ocean bathymetry (depth normalized in [-2000.0, 0.0] meters)
        - Channel 3: Continuous distance to shore (normalized in [0.0, 50000.0] meters)
        - Channel 4: Ocean wind speed (speed normalized in [0.0, 25.0] m/s)
    Model Target: (Batch_Size, 2, 256, 256) float32 in [0.0, 1.0]
        - Channel 0: Mobile Vessel Centroid Gaussian peaks (sigma = 2.0 px)
        - Channel 1: Fixed Offshore Structure Centroid Gaussian peaks (sigma = 2.0 px)
"""

import os
import math
from typing import Tuple, List, Dict, Optional
import numpy as np
import pandas as pd
import tifffile
import cv2
from scipy.ndimage import distance_transform_edt
import torch
from torch.utils.data import Dataset, DataLoader

from globals import (
    PATCH_SIZE, INPUT_CHANNELS, NUM_CLASSES,
    VH_MIN_DB, VH_MAX_DB, VV_MIN_DB, VV_MAX_DB,
    MAX_DISTANCE_SHORE_METERS, MAX_BATHYMETRY_METERS, MAX_WIND_SPEED_MS, NODATA_VALUE,
    GAUSSIAN_RADIAL_SIGMA, BATCH_SIZE, NUM_WORKERS,
    TRAIN_SCENES, VAL_SCENES, TEST_SCENES, DATA_DIR, LABELS_PATH
)


# -----------------------------------------------------------------------------
# 1. Radiometric and Geospatial Normalization Utilities
# -----------------------------------------------------------------------------
def normalize_sar_db(arr: np.ndarray, min_db: float, max_db: float) -> np.ndarray:
    """
    Standardizes raw SAR backscatter values (decibels) into [0.0, 1.0].
    Substitutes sensor nodata (-32768.0) and values below the noise floor with min_db.
    """
    # pixels on th edges of the orbital acquisition | NAN/corrupted pixels | pixels under the NESZ threshold
    clean_arr = np.where((arr == NODATA_VALUE) | np.isnan(arr) | (arr < min_db), min_db, arr) # extract valid data
    # the 1 value needs to be fixed and not a scene specific maximum because it would
    # introduce Data Shift between scenes
    norm = (clean_arr - min_db) / (max_db - min_db) # (actual values) / (new max value)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def normalize_bathymetry(depth: np.ndarray) -> np.ndarray:
    """
    Normalizes ocean bathymetry: depths up to -2000m mapped into [0.0, 1.0].
    Emerged land (depth >= 0.0) and nodata are mapped to 0.0.
    """
    clean_depth = np.where((depth == NODATA_VALUE) | np.isnan(depth) | (depth >= 0.0), 0.0, depth)
    norm = clean_depth / MAX_BATHYMETRY_METERS  # MAX_BATHYMETRY_METERS = -2000.0
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def normalize_distance_to_shore(dist_m: np.ndarray) -> np.ndarray:
    """
    Normalizes geodetic distance from coastline: metric distances up to 50 km scaled to [0.0, 1.0].
    """
    norm = dist_m / MAX_DISTANCE_SHORE_METERS  # MAX_DISTANCE_SHORE_METERS = 50000.0
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def normalize_wind_speed(wind: np.ndarray, max_wind: float = MAX_WIND_SPEED_MS) -> np.ndarray:
    """
    Standardizes ocean wind speed (m/s) into [0.0, 1.0].
    Substitutes sensor nodata (-32768.0), NaN, and negative values with 0.0.
    Clips high wind speeds to max_wind (25.0 m/s).
    """
    clean_wind = np.where((wind == NODATA_VALUE) | np.isnan(wind) | (wind < 0.0), 0.0, wind)
    # normalization to 25m/s because it is a 10 on the Beaufort scale
    norm = clean_wind / max_wind # if max_wind is lower than the actual max the values greater than max_wind get clipped to 1
    return np.clip(norm, 0.0, 1.0).astype(np.float32)



# -----------------------------------------------------------------------------
# 2. Continuous Gaussian Target Heatmap Generator (CenterNet)
# -----------------------------------------------------------------------------
def draw_gaussian_peak(
    heatmap: np.ndarray,
    center_x: int,
    center_y: int,
    sigma: float = GAUSSIAN_RADIAL_SIGMA
) -> None:
    """
    Renders a 2D Gaussian peak centered at (center_x, center_y) directly into
    the provided heatmap array (H, W). Merges overlaps using pointwise maximum.
    """
    H, W = heatmap.shape
    # using the 3sigma rule we calculate the gaussian only on a radius*2xradius*2 subgrid
    radius = int(3 * sigma)

    x0 = max(0, center_x - radius)
    x1 = min(W, center_x + radius + 1) # +1 to include c_x + r
    y0 = max(0, center_y - radius)
    y1 = min(H, center_y + radius + 1)

    if x0 >= x1 or y0 >= y1: # happens only if the gaussian is completely out of the heatmap
        return

    # vectors containing the coordinates in the grid over the subgrid stands
    grid_y, grid_x = np.ogrid[y0:y1, x0:x1]
    # square euclidean distance between each pixel and the center
    dist_sq = (grid_x - center_x) ** 2 + (grid_y - center_y) ** 2 
    # calculate bidimensional gaussian distribution
    gaussian = np.exp(-dist_sq / (2.0 * sigma * sigma)).astype(np.float32)

    # getting the maximum and not summing prevents going over 1 in the probability (centernet paradigm)
    heatmap[y0:y1, x0:x1] = np.maximum(heatmap[y0:y1, x0:x1], gaussian)


# -----------------------------------------------------------------------------
# 3. Geospatial Scene Context Cache (Bathymetry & Shore Distance via EDT)
# -----------------------------------------------------------------------------
class SceneContextCache:
    """
    Caches low-resolution environmental rasters (bathymetry, owiMask, owiWindSpeed, ~441x596, ~500 KB)
    and computes the continuous distance-to-shore field on-the-fly using Euclidean
    Distance Transform (EDT).
    """
    def __init__(self, scene_id: str, data_dir: str = DATA_DIR):
        scene_dir = os.path.join(data_dir, scene_id)
        bathy_path = os.path.join(scene_dir, "bathymetry.tif")
        owi_path = os.path.join(scene_dir, "owiMask.tif")
        wind_path = os.path.join(scene_dir, "owiWindSpeed.tif")

        # read raw bathymetry data
        bathy_raw = tifffile.imread(bathy_path).astype(np.float32)
        self.bathy_norm = normalize_bathymetry(bathy_raw)

        # In Sentinel-1 OWI products: 0 = useful ocean, 1 = land, 2 = ice.
        owi_raw = tifffile.imread(owi_path)
        is_ocean = (owi_raw == 0) # we change the 0 pixels to True (1) and the rest False (0)
        # EDT computes distance from non-zero pixels to the nearest zero pixel (land/ice).
        dist_pixels = distance_transform_edt(is_ocean)
        dist_meters = (dist_pixels * 500.0).astype(np.float32)  # ~500m/pixel spatial resolution

        # ensure identical spatial grid dimension between bathymetry and EDT shore distance
        if dist_meters.shape != self.bathy_norm.shape:
            dist_meters = cv2.resize(
                dist_meters,
                (self.bathy_norm.shape[1], self.bathy_norm.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        self.dist_shore_norm = normalize_distance_to_shore(dist_meters)

        # OWI wind speed (~500 m/px native) -> normalized, aligned to the context grid
        wind_raw = tifffile.imread(wind_path).astype(np.float32)
        self.wind_norm = normalize_wind_speed(wind_raw)
        if self.wind_norm.shape != self.bathy_norm.shape:
            self.wind_norm = cv2.resize(
                self.wind_norm,
                (self.bathy_norm.shape[1], self.bathy_norm.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )
        # we define the size of the context window as the size of the bathymetry map
        self.context_h, self.context_w = self.bathy_norm.shape

    def extract_context_patch(
        self,
        r0: int,
        r1: int,
        c0: int,
        c1: int,
        sar_h: int,
        sar_w: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts and bilinearly interpolates the corresponding context window
        to high-resolution (PATCH_SIZE, PATCH_SIZE).
        """
        # context data and radio data have different resolution so they need to be scaled accordingly
        scale_r = self.context_h / float(sar_h)
        scale_c = self.context_w / float(sar_w)

        # getting the angles of the context window related to the patch
        cr0 = max(0, int(r0 * scale_r))
        cr1 = min(self.context_h, int(math.ceil(r1 * scale_r)) + 1)
        cc0 = max(0, int(c0 * scale_c))
        cc1 = min(self.context_w, int(math.ceil(c1 * scale_c)) + 1)

        # getting the actual data from the various context maps
        b_sub = self.bathy_norm[cr0:cr1, cc0:cc1]
        d_sub = self.dist_shore_norm[cr0:cr1, cc0:cc1]
        w_sub = self.wind_norm[cr0:cr1, cc0:cc1]

        # guard against zero-area slices on edge borders (we need at least 4 angles, not present on only one pixel)
        if b_sub.shape[0] < 2 or b_sub.shape[1] < 2:
            cr1 = min(self.context_h, cr0 + 2)
            cc1 = min(self.context_w, cc0 + 2)
            b_sub = self.bathy_norm[cr0:cr1, cc0:cc1]
            d_sub = self.dist_shore_norm[cr0:cr1, cc0:cc1]
            w_sub = self.wind_norm[cr0:cr1, cc0:cc1]

        # resizing to the size of the SAR patch
        b_patch = cv2.resize(b_sub, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_LINEAR)
        d_patch = cv2.resize(d_sub, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_LINEAR)
        w_patch = cv2.resize(w_sub, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_LINEAR)
        return b_patch, d_patch, w_patch


# -----------------------------------------------------------------------------
# 4. PyTorch Dataset for Dark Vessel Detection
# -----------------------------------------------------------------------------
class DarkVesselDataset(Dataset):
    """
    PyTorch Dataset for Sentinel-1 Dark Vessel Detection with CGCAM.
    Extracts balanced 256x256 chips (50% positive targets with jitter, 50% background negatives).
    """
    def __init__(
        self,
        scene_ids: List[str],
        data_dir: str = DATA_DIR,
        labels_path: str = LABELS_PATH,
        is_train: bool = True
    ):
        self.scene_ids = scene_ids
        self.data_dir = data_dir
        self.is_train = is_train

        # Load annotations filtered by selected scene IDs
        labels_df = pd.read_csv(labels_path)
        self.df = labels_df[labels_df["scene_id"].isin(scene_ids)].copy()

        # Cache file paths, shapes, and context per scene
        self.sar_paths: Dict[str, Dict[str, str]] = {}
        self.scene_shapes: Dict[str, Tuple[int, int]] = {}
        self.context_cache: Dict[str, SceneContextCache] = {}
        self.scene_targets: Dict[str, np.ndarray] = {}

        for sid in scene_ids:
            s_dir = os.path.join(data_dir, sid)
            vh_path = os.path.join(s_dir, "VH_dB.tif")
            vv_path = os.path.join(s_dir, "VV_dB.tif")
            self.sar_paths[sid] = {"VH": vh_path, "VV": vv_path}

            # Inspect shape without keeping file descriptor open across worker processes
            with tifffile.TiffFile(vh_path) as tif:
                self.scene_shapes[sid] = tif.series[0].shape

            self.context_cache[sid] = SceneContextCache(sid, data_dir=data_dir)

            # Pre-group targets for sub-millisecond spatial querying
            sdf = self.df[self.df["scene_id"] == sid]
            if len(sdf) > 0:
                rows = sdf["detect_scene_row"].values.astype(np.int32)
                cols = sdf["detect_scene_column"].values.astype(np.int32)
                # is_vessel: False -> 0 (infrastructure), True or NaN -> 1 (vessel)
                is_v = np.where(sdf["is_vessel"] == False, 0, 1).astype(np.int32)
                self.scene_targets[sid] = np.stack([rows, cols, is_v], axis=1)
            else:
                self.scene_targets[sid] = np.empty((0, 3), dtype=np.int32)

        # Lazy memmap dictionary initialized per worker process
        self._sar_memmaps: Optional[Dict[str, Dict[str, np.memmap]]] = None

        # Build list of sample coordinates
        self.samples = self._build_samples()

    def _get_sar_memmap(self, sid: str, pol: str) -> np.memmap:
        """Lazily creates and caches numpy memory-mapped views per worker process."""
        if self._sar_memmaps is None:
            self._sar_memmaps = {}
        if sid not in self._sar_memmaps:
            self._sar_memmaps[sid] = {}
        if pol not in self._sar_memmaps[sid]:
            self._sar_memmaps[sid][pol] = tifffile.memmap(self.sar_paths[sid][pol])
        return self._sar_memmaps[sid][pol]

    def _build_samples(self) -> List[Dict]:
        """Constructs balanced positive and negative sampling index."""
        samples = []
        half = PATCH_SIZE // 2

        # 1. Positive Samples: centered on each ground-truth target
        for _, row in self.df.iterrows():
            sid = row["scene_id"]
            H, W = self.scene_shapes[sid]
            r_center = int(row["detect_scene_row"])
            c_center = int(row["detect_scene_column"])

            # Ensure patch falls strictly within scene boundaries
            r0 = max(0, min(H - PATCH_SIZE, r_center - half))
            c0 = max(0, min(W - PATCH_SIZE, c_center - half))

            samples.append({
                "scene_id": sid,
                "r0": r0,
                "c0": c0,
                "is_positive": True,
            })

        # 2. Negative Samples (50:50 ratio): sampled across ocean and coastal regions
        num_neg = len(samples)
        rng = np.random.RandomState(42 if not self.is_train else None)
        for _ in range(num_neg):
            sid = rng.choice(self.scene_ids)
            H, W = self.scene_shapes[sid]
            r0 = rng.randint(0, H - PATCH_SIZE)
            c0 = rng.randint(0, W - PATCH_SIZE)
            samples.append({
                "scene_id": sid,
                "r0": r0,
                "c0": c0,
                "is_positive": False,
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        sid = sample["scene_id"]
        H_sar, W_sar = self.scene_shapes[sid]

        r0, c0 = sample["r0"], sample["c0"]

        # In training mode for positive samples, apply random spatial jitter (+- 32 px)
        if self.is_train and sample["is_positive"]:
            jitter_r = np.random.randint(-32, 33)
            jitter_c = np.random.randint(-32, 33)
            r0 = max(0, min(H_sar - PATCH_SIZE, r0 + jitter_r))
            c0 = max(0, min(W_sar - PATCH_SIZE, c0 + jitter_c))

        r1 = r0 + PATCH_SIZE
        c1 = c0 + PATCH_SIZE

        # 1. Fast sub-window extraction via memory mapping
        vh_mmap = self._get_sar_memmap(sid, "VH")
        vv_mmap = self._get_sar_memmap(sid, "VV")

        vh_raw = np.array(vh_mmap[r0:r1, c0:c1], dtype=np.float32)
        vv_raw = np.array(vv_mmap[r0:r1, c0:c1], dtype=np.float32)

        # 2. Radiometric SAR normalization to [0.0, 1.0]
        vh_norm = normalize_sar_db(vh_raw, VH_MIN_DB, VH_MAX_DB)
        vv_norm = normalize_sar_db(vv_raw, VV_MIN_DB, VV_MAX_DB)

        # 3. Context patch extraction (bathymetry, distance-to-shore, wind-speed)
        bathy_patch, dist_shore_patch, wind_patch = self.context_cache[sid].extract_context_patch(
            r0, r1, c0, c1, H_sar, W_sar
        )

        # Assemble 5-channel input tensor: shape (5, 256, 256)
        x = np.stack([vh_norm, vv_norm, bathy_patch, dist_shore_patch, wind_patch], axis=0)

        # 4. Target Heatmap generation: shape (2, 256, 256)
        target = np.zeros((NUM_CLASSES, PATCH_SIZE, PATCH_SIZE), dtype=np.float32)

        targets = self.scene_targets[sid]
        if len(targets) > 0:
            mask = (
                (targets[:, 0] >= r0) & (targets[:, 0] < r1) &
                (targets[:, 1] >= c0) & (targets[:, 1] < c1)
            )
            patch_targets = targets[mask]
            for r_t, c_t, is_v in patch_targets:
                local_y = int(r_t - r0)
                local_x = int(c_t - c0)
                ch = 0 if is_v == 1 else 1
                draw_gaussian_peak(target[ch], local_x, local_y, sigma=GAUSSIAN_RADIAL_SIGMA)

        # 5. SAR-Safe equivariant geometric augmentations (training only)
        if self.is_train:
            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=2).copy()
                target = np.flip(target, axis=2).copy()
            # Random Vertical Flip
            if np.random.rand() > 0.5:
                x = np.flip(x, axis=1).copy()
                target = np.flip(target, axis=1).copy()
            # Random 90-degree Rotations (SAR geometry is nadir-looking top-down)
            rot_k = np.random.choice([0, 1, 2, 3])
            if rot_k > 0:
                x = np.rot90(x, k=rot_k, axes=(1, 2)).copy()
                target = np.rot90(target, k=rot_k, axes=(1, 2)).copy()

        return torch.from_numpy(x).float(), torch.from_numpy(target).float()


# -----------------------------------------------------------------------------
# 5. DataLoader Factory
# -----------------------------------------------------------------------------
def get_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    data_dir: str = DATA_DIR,
    labels_path: str = LABELS_PATH
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs and returns PyTorch DataLoaders for Training, Validation, and Testing.
    Enforces scene-level partitioning strictly avoiding spatial autocorrelation.
    """
    train_dataset = DarkVesselDataset(
        scene_ids=TRAIN_SCENES,
        data_dir=data_dir,
        labels_path=labels_path,
        is_train=True
    )
    val_dataset = DarkVesselDataset(
        scene_ids=VAL_SCENES,
        data_dir=data_dir,
        labels_path=labels_path,
        is_train=False
    )
    test_dataset = DarkVesselDataset(
        scene_ids=TEST_SCENES,
        data_dir=data_dir,
        labels_path=labels_path,
        is_train=False
    )

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("Testing DarkVesselDataset & DataLoader pipeline contract...")
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=4, num_workers=0)
    print(f"Dataset partition sizes -> Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    x, y = next(iter(train_loader))
    print(f"Batch Input Tensor shape:  {x.shape} (Expected: [4, 5, 256, 256])")
    print(f"Batch Target Tensor shape: {y.shape} (Expected: [4, 2, 256, 256])")
    print(f"Input channels range:")
    channel_names = ["VH_dB", "VV_dB", "bathymetry", "distance_to_shore", "wind_speed"]
    for c_idx, name in enumerate(channel_names):
        print(f"  Ch {c_idx} ({name}): min={x[:, c_idx].min().item():.4f}, max={x[:, c_idx].max().item():.4f}, mean={x[:, c_idx].mean().item():.4f}")
    print(f"Target channels range:")
    target_names = ["Mobile Vessel", "Fixed Offshore Structure"]
    for t_idx, name in enumerate(target_names):
        print(f"  Ch {t_idx} ({name}): min={y[:, t_idx].min().item():.4f}, max={y[:, t_idx].max().item():.4f}, max_peak={y[:, t_idx].max().item():.4f}")

    assert x.shape == (4, 5, 256, 256), f"Input shape mismatch: {x.shape}"
    assert y.shape == (4, 2, 256, 256), f"Target shape mismatch: {y.shape}"
    assert x.min() >= 0.0 and x.max() <= 1.0, f"Input outside [0.0, 1.0]: [{x.min()}, {x.max()}]"
    assert y.min() >= 0.0 and y.max() <= 1.0, f"Target outside [0.0, 1.0]: [{y.min()}, {y.max()}]"
    print("Data pipeline contract verified successfully!")
