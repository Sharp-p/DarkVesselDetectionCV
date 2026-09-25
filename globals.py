"""
globals.py

Global configurations, hyperparameters, device mapping, deterministic random seed,
and contract tensor definitions for Dark Vessel Detection via Multimodal SAR and CGCAM.
Architecture Paradigm: Unified 2-Channel Anchor-Free CenterNet (Detection + Classification).
"""

import os
import torch
import random
import numpy as np

# ---------------------------------------------------------
# 1. Deterministic Reproducibility & Device Configuration
# ---------------------------------------------------------
RANDOM_SEED = 42

def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)

# ---------------------------------------------------------
# 2. File System Paths & Scene Partitioning (Scene-Level)
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "..", "downloaded")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Scene-level partitioning strictly avoiding spatial autocorrelation
TRAIN_SCENES = [
    "05bc615a9b0e1159t",
    "2899cfb18883251bt",
    "72dba3e82f782f67t",
    "cbe4ad26fe73f118t",
]
VAL_SCENES = [
    "e98ca5aba8849b06t",
]
TEST_SCENES = [
    "590dd08f71056cacv",
    "b1844cde847a3942v",
]

# ---------------------------------------------------------
# 3. Radiometric & Geospatial Calibration Constants
# ---------------------------------------------------------
VH_MIN_DB = -35.0
VH_MAX_DB = 10.0
VV_MIN_DB = -30.0
VV_MAX_DB = 15.0
MAX_DISTANCE_SHORE_METERS = 50000.0  # 50 km normalization scale
MAX_BATHYMETRY_METERS = -2000.0      # 2000 m ocean depth normalization
NODATA_VALUE = -32768.0

# ---------------------------------------------------------
# 4. Patch & Model Input/Output Tensor Contracts (Option 1)
# ---------------------------------------------------------
PATCH_SIZE = 256    # H = W = 256
INPUT_CHANNELS = 4  # [0: VH_dB, 1: VV_dB, 2: bathymetry, 3: distance_to_shore]
NUM_CLASSES = 2     # [0: Mobile Vessel, 1: Fixed Offshore Structure]

# Expected Batch Tensor Shapes:
# Model Input:   (Batch_Size, 4, PATCH_SIZE, PATCH_SIZE)
# Model Output:  (Batch_Size, 2, PATCH_SIZE, PATCH_SIZE)
#   - Channel 0: Mobile Vessel Centroid Gaussian peaks in [0.0, 1.0]
#   - Channel 1: Fixed Offshore Structure Centroid Gaussian peaks in [0.0, 1.0]

# ---------------------------------------------------------
# 5. Training Hyperparameters & Focal Loss Configuration
# ---------------------------------------------------------
BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
NUM_EPOCHS = 30

# CenterNet Modified Focal Loss Parameters
FOCAL_ALPHA = 2.0            # Modulating exponent suppressing easy ocean background
FOCAL_BETA = 4.0             # Gaussian penalty exponent softening near-miss predictions
GAUSSIAN_RADIAL_SIGMA = 2.0  # Radius of ground-truth target Gaussian peaks in pixels

# Peak Detection & Evaluation Thresholds
PEAK_CONFIDENCE_THRESHOLD = 0.3  # Minimum heatmap activation to declare a detection
NMS_KERNEL_SIZE = 3              # Local max-pooling window (3x3 pixels) for peak extraction
MATCH_DISTANCE_PIXELS = 10       # Tolerance radius (100 meters at 10m/pixel) for true positives

# ---------------------------------------------------------
# 6. Novelty & Ablation Study Settings
# ---------------------------------------------------------
USE_CGCAM = True  # Toggle True (Proposed CGCAM) vs False (Baseline SAR-Only)
