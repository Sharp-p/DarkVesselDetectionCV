# Project Specification & Execution Plan: Dark Vessel Detection via Multimodal SAR and Context-Guided Cross-Attention (CGCAM)

**Course:** Computer Vision (A.Y. 2025–2026) — Sapienza University of Rome  
**Laboratory:** ALCOR Lab, Department of Computer, Control and Management Engineering (DIAG)  
**Instructor:** Prof. Irene Amerini  
**Teaching Assistants:** Francesco Pro, Gianmarco Scarano, Simone Teglia  
**Project Repository:** `DarkVesselDetectionCV`  

---

## Table of Contents
1. [Course & Examination Framework](#1-course--examination-framework)
   - [Grade Composition & Session Rules](#11-grade-composition--session-rules)
   - [Project Submission Deadlines & Deliverables](#12-project-submission-deadlines--deliverables)
   - [Presentation Constraints & Slide Layout](#13-presentation-constraints--slide-layout)
2. [Domain Context & Theoretical Foundations](#2-domain-context--theoretical-foundations)
   - [Problem Statement: Dark Vessels & IUU Fishing](#21-problem-statement-dark-vessels--iuu-fishing)
   - [SAR Physical Principles & Microwave Scattering](#22-sar-physical-principles--microwave-scattering)
   - [Dual Polarization: VV vs. VH Channels](#23-dual-polarization-vv-vs-vh-channels)
   - [Multiplicative Speckle Noise Model](#24-multiplicative-speckle-noise-model)
3. [Dataset Pipeline & Geospatial Engineering](#3-dataset-pipeline--geospatial-engineering)
   - [Dataset Overview: xView3-SAR and SARFish](#31-dataset-overview-xview3-sar-and-sarfish)
   - [Critique of the Competition Split & Spatial Leakage Mitigation](#32-critique-of-the-competition-split--spatial-leakage-mitigation)
   - [Scene-Level Partitioning Strategy](#33-scene-level-partitioning-strategy)
   - [GeoTIFF Float16 Radiometric Preprocessing & Tiling](#34-geotiff-float16-radiometric-preprocessing--tiling)
4. [Methodological Novelty: CGCAM Architecture](#4-methodological-novelty-cgcam-architecture)
   - [Architectural Rationale](#41-architectural-rationale)
   - [Context-Guided Cross-Attention Module (CGCAM) Formulation](#42-context-guided-cross-attention-module-cgcam-formulation)
   - [Multi-Task Prediction Heads](#43-multi-task-prediction-heads)
5. [Three-Member Collaborative Division of Labor](#5-three-member-collaborative-division-of-labor)
   - [Member 1: Data & Geophysics Engineer](#51-member-1-data--geophysics-engineer)
   - [Member 2: Model & Novelty Architect](#52-member-2-model--novelty-architect)
   - [Member 3: Training, Evaluation & Ablation Specialist](#53-member-3-training-evaluation--ablation-specialist)
   - [Parallel Decoupling via Mock Tensors & Git Workflow](#54-parallel-decoupling-via-mock-tensors--git-workflow)
6. [Contract Interface Specification (`globals.py`)](#6-contract-interface-specification-globalspy)
7. [Ablation Study & Experimental Protocol](#7-ablation-study--experimental-protocol)
8. [Final Presentation Breakdown (10 Minutes / 10 Slides)](#8-final-presentation-breakdown-10-minutes--10-slides)

---

## 1. Course & Examination Framework

### 1.1 Grade Composition & Session Rules
The final evaluation is mathematically defined as:
$$\text{Final Grade} = 0.5 \cdot \text{Written Exam} + 0.5 \cdot \text{Project Assessment}$$
where both components must satisfy $\ge 18/30$.
* *Conceptual Meaning:* An equal 50%/50% arithmetic balance between theoretical mastery (1-hour written exam on classical CV, transforms, geometry, and representations) and practical engineering competence (PyTorch implementation, architectural novelty, and 10-minute oral defense).


* **Separation of Components:** The written examination and the project presentation can be taken across different exam sessions. If only one component is taken during a specific session, an administrative mark of *"refused/rinuncia"* is recorded on Infostud to preserve the partial grade.
* **Academic Year Validity:** Both parts must be finalized within the same academic year (between June 2026 and March 2027).
* **Framework Constraint:** All deep learning components must be implemented strictly using **PyTorch**. No external monolithic frameworks or wrappers that obscure the underlying architecture are permitted.
* **Code Architecture:** The repository must strictly adhere to the standardized 6-module architecture:
  `globals.py`, `utils.py`, `data.py`, `network.py`, `train.py`, `evaluation.py`.

### 1.2 Project Submission Deadlines & Deliverables
To register for the project presentation, two strict deadlines must be observed:

| Deadline | Temporal Window | Action Required |
| :--- | :--- | :--- |
| **Registration Form** | $\ge \mathbf{7\text{ days}}$ prior to presentation | Submit the official Google Form specifying student names, IDs, institutional emails (`@studenti.uniroma1.it`), chosen topic, and the public GitHub repository link. |
| **Code & Slides Freeze** | $\mathbf{2\text{ days}}$ prior to presentation | Final commits to the GitHub repository. After this timestamp, the codebase and slide decks are considered frozen for faculty evaluation. |

**Mandatory Repository Contents:**
1. Clean, modularized PyTorch implementation following the 6 required modules.
2. Presentation slides in `.pdf` and/or `.pptx` format.
3. A comprehensive `README.md` containing an architectural overview, dependency requirements, and exact execution commands to reproduce both training and evaluation pipelines.
4. Dataset download instructions or direct ingestion scripts.

### 1.3 Presentation Constraints & Slide Layout
* **Duration:** Strictly **10 minutes** total, followed by oral defense and individual technical questions from the committee.
* **Visual Standards:**
  * Clean **white background** across all slides.
  * High-contrast graphical elements, plots, and figures.
  * Mandatory slide pagination displayed in the format $\mathbf{X/Y}$ (Current Slide / Total Slides) on every single page.
* **Physical Backup:** The slide deck must be uploaded to GitHub and brought to the exam room on a **USB flash drive**.

---

## 2. Domain Context & Theoretical Foundations

### 2.1 Problem Statement: Dark Vessels & IUU Fishing
Maritime surveillance is essential for biodiversity conservation and border security. Commercial vessels are mandated by the International Maritime Organization (IMO) to broadcast Automatic Identification System (AIS) transponder signals. However, vessels engaged in **Illegal, Unreported, and Unregulated (IUU) fishing**, smuggling, or sanctions evasion deliberately disable their AIS transponders, becoming **"dark vessels"**. 

Optical satellite imagery (e.g., Sentinel-2) fails during overcast weather, fog, and nighttime. Spaceborne **Synthetic Aperture Radar (SAR)** (e.g., Sentinel-1) provides continuous, all-weather, day-and-night active microwave monitoring, making it the primary modality for maritime dark vessel detection.

### 2.2 SAR Physical Principles & Microwave Scattering
SAR synthesizes a virtual antenna aperture of length $L_{\text{syn}} \gg L$ through the orbital motion of the satellite:
* **Azimuth Resolution:** Completely independent of slant range distance $R_0$ and wavelength $\lambda$:
  $$\delta_{az,\text{SAR}} = \frac{L}{2}$$
  *Conceptual Meaning:* By recording Doppler phase histories as the satellite travels, SAR synthesizes an antenna kilometers long. The spatial resolution along the flight path depends strictly on half the physical antenna length $L$ ($L \approx 12.3\text{ m} \implies \delta_{az} \approx 6.1\text{ m}$), eliminating orbital altitude degradation.
* **Slant Range & Ground Range Resolution:** Determined by the bandwidth $B$ of the frequency-modulated chirp:
  $$\delta_r = \frac{c}{2B}, \quad \delta_{gr} = \frac{\delta_r}{\sin \theta_i} = \frac{c}{2B \sin \theta_i}$$
  *Conceptual Meaning:* The radial resolution $\delta_r$ along the radar line of sight depends on the speed of light $c$ and chirp frequency bandwidth $B$. Projecting this onto the horizontal Earth surface via the local incidence angle $\theta_i$ yields the physical ground resolution $\delta_{gr}$.

### 2.3 Dual Polarization: VV vs. VH Channels
Sentinel-1 operates in dual polarization, transmitting Vertical ($V$) waves and receiving both Vertical ($V$) and Horizontal ($H$) returns:
* **$VV$ Channel (Co-Polarized):** Governed by resonant **Bragg scattering** over water ripples. Strongly modulated by wind velocity and sea-state roughness ($\sigma^0 \in [-20\text{ dB}, -8\text{ dB}]$ in rough waters).
* **$VH$ Channel (Cross-Polarized):** Depolarization occurs due to **multiple-bounce dihedral and trihedral reflections** occurring between the vertical metallic hull plates and the horizontal sea surface. Bragg scattering from sea surface does not rotate polarization; thus, the ocean clutter remains near the sensor noise floor ($\sigma^0 \in [-35\text{ dB}, -28\text{ dB}]$), while ship hulls reflect strongly ($\sigma^0 \in [0\text{ dB}, +15\text{ dB}]$).
* **Target-to-Clutter Ratio (TCR):**
  $$\text{TCR} = \frac{\sigma^0_{\text{target}}}{\sigma^0_{\text{clutter}}} \implies \text{TCR}_{VH} \gg \text{TCR}_{VV}$$
  *Conceptual Meaning:* Measures the contrast of the metallic vessel relative to the surrounding sea. In $VV$, rough water causes high clutter (low TCR). In $VH$, the sea surface does not depolarize waves (near-zero clutter), while multiple dihedral reflections off metallic hulls rotate polarization, yielding an extraordinarily high TCR.

### 2.4 Multiplicative Speckle Noise Model
Due to the coherent superposition of sub-resolution scatterers within each resolution cell, SAR imagery is intrinsically corrupted by **multiplicative speckle noise**:
$$I(x, y) = R(x, y) \cdot \eta(x, y)$$
where $R(x, y) \in \mathbb{R}^+$ is the underlying terrain radar cross-section, and $\eta(x, y) \sim \Gamma(L, 1/L)$ is a Gamma-distributed stochastic noise component with unit mean and variance $1/L$.
*Conceptual Meaning:* Unlike additive sensor noise ($I = R + \mathcal{N}$), speckle noise scales multiplicatively with signal intensity $R$. Bright areas fluctuate more violently, misleading standard gradient-based edge detectors (Sobel, Canny) and causing false object detections on wind-driven ocean waves.

**Impact on Computer Vision:**
1. Gradient-based operators (Sobel, Canny) compute false edges across open water.
2. Standard object detectors (YOLO, Faster R-CNN) mistake wind-induced wave crests for small fishing vessels, yielding high false positive rates.
3. Coastal terrain exhibits severe geometric distortions (**layover** and **foreshortening**), where cliff faces appear brighter than ship hulls.

---

## 3. Dataset Pipeline & Geospatial Engineering

### 3.1 Dataset Overview: xView3-SAR and SARFish
* **Source:** U.S. Defense Innovation Unit (DIU), Global Fishing Watch, and DSTG (Australia).
* **Sensors:** Sentinel-1 C-band SAR ($5.54\text{ cm}$ wavelength, $\approx 10\text{ m}$ spatial resolution).
* **Channels per Scene:**
  1. `VH_dB.tif`: Cross-polarized radar backscatter in decibels.
  2. `VV_dB.tif`: Co-polarized radar backscatter in decibels.
  3. `bathymetry.tif`: Ocean floor depth in meters (GEBCO bathymetric model).
  4. `distance_to_shore.tif`: Metric distance of each pixel to the nearest coastline.
  5. `owiWindSpeed.tif` / `owiWindDirection.tif`: Ocean surface wind vectors.
* **Ground Truth Annotations:**
  `scene_id`, `detect_scene_row`, `detect_scene_column`, `is_vessel` (boolean), `is_fishing` (boolean), `vessel_length_m` (continuous float), `confidence` (LOW, MEDIUM, HIGH).

### 3.2 Critique of the Competition Split & Spatial Leakage Mitigation
The original xView3 competition partition comprised 500 training scenes, 50 validation scenes, and 150 hidden leaderboard scenes (~2 TB total).
* **Impracticability:** Ingesting 500 scenes ($29.783 \times 22.081$ pixels each) is computationally redundant for architectural research and impossible on workstation resources.
* **Spatial Autocorrelation (Spatial Leakage):** Slicing patches randomly across scenes into train and test sets leads to catastrophic data leakage. Patches from the same satellite swath share identical sensor incident angles, orbital geometry, wind velocity, and ambient sea clutter.
* **Academic Soundness:** The model must be evaluated on **unseen geographic swaths** to demonstrate true out-of-distribution domain generalization.

### 3.3 Scene-Level Partitioning Strategy
Utilizing the 7-scene package (~8.2 GB uncompressed, verified intact in `downloaded/`):
* **Training Set ($\sim 70\%$):** 4 scenes (`05bc615a9b0e1159t`, `2899cfb18883251bt`, `72dba3e82f782f67t`, `cbe4ad26fe73f118t`).
* **Validation Set ($\sim 15\%$):** 1 scene (`e98ca5aba8849b06t`) used strictly for loss tracking, early stopping, and hyperparameter tuning.
* **Test Set ($\sim 15-20\%$):** 2 scenes (`590dd08f71056cacv`, `b1844cde847a3942v`) frozen until final inference and ablation benchmarks.

### 3.4 GeoTIFF Float16 Radiometric Preprocessing & Tiling
* **Encoding:** IEEE 754 `Float16` (`SampleFormat=3`, `BitsPerSample=16`, `GDAL_DATA_TYPE=15`).
* **Memory Management:** Memory-mapped streaming via `numpy.memmap` reading contiguous strip offsets without loading the 1.3 GB file into RAM.
* **Radiometric Standardization:**
  $$\sigma^0_{[\text{norm}]} = \text{clip}\left(\frac{\sigma^0_{[\text{dB}]} - \sigma^0_{\min}}{\sigma^0_{\max} - \sigma^0_{\min}}, 0.0, 1.0\right), \quad \sigma^0_{\min} = -35.0\text{ dB}, \ \sigma^0_{\max} = +10.0\text{ dB}$$
  *Conceptual Meaning:* Standardizes radar decibel values into $[0.0, 1.0]$ for stable neural network training. Values below $-35\text{ dB}$ (thermal sensor noise) and above $+10\text{ dB}$ (extreme specular metallic saturation) are clamped, preventing gradient explosion. Nodata values ($-32768.0$) are masked to $0.0$.
* **Context Normalization:**
  $$\text{DistShore}_{[\text{norm}]} = \text{clip}\left(\frac{D}{50000.0}, 0.0, 1.0\right), \quad \text{Bathy}_{[\text{norm}]} = \text{clip}\left(\frac{\text{depth}}{-2000.0}, 0.0, 1.0\right)$$
  *Conceptual Meaning:* Distances beyond $50\text{ km}$ ($50000\text{ m}$) saturate to $1.0$ (deep open sea), concentrating gradient sensitivity on coastal waters where false alarms occur. Similarly, bathymetric depths below $-2000\text{ m}$ saturate to $1.0$, preserving model sensitivity for continental shelves and navigable shallows.
* **Balanced Tiling Scheme:**
  - Patch resolution: $256 \times 256$ pixels.
  - Positive patches: Centered on ground truth vessels with random jittering ($\pm 32$ pixels).
  - Negative patches: Hard negative mining over coastal cliffs and high-wind wave regions (50:50 positive-to-negative ratio).

---

## 4. Methodological Novelty: CGCAM Architecture

### 4.1 Architectural Rationale
Standard object detection frameworks concatenate contextual channels (bathymetry, distance to coast) directly into early convolutional layers alongside radar channels. Because coastal cliffs exhibit high radar intensity and complex textures, early convolution filters confuse cliff rocks with ships. 

We propose decoupling radar representation from geophysical priors, using **Context-Guided Cross-Attention Module (CGCAM)**: auxiliary geophysical channels modulate and filter radar feature maps at bottleneck representations.

```
[ SAR VV, VH ] ------------> Backbone CNN ---------> Radar Features (F_r)
                                                           |
                                                           v
[ Bathymetry, DistCoast ] -> Context Encoder --------> [ CGCAM Module ] ---> Refined Features (F_out)
                                                       (Cross-Attention)               |
                                                                                       v
                                                                          [ Decoder / Upsampling Neck ]
                                                                                       |
                                                                                       v
                                                                         Unified 2-Channel Heatmap Head
                                                                              (B, 2, 256, 256)
                                                                         Ch 0: Mobile Vessel Centroids
                                                                         Ch 1: Fixed Offshore Structures
```

### 4.2 Context-Guided Cross-Attention Module (CGCAM) Formulation
Let $\mathbf{F}_r \in \mathbb{R}^{C \times H \times W}$ be the deep feature map extracted from the radar backbone, and $\mathbf{F}_c \in \mathbb{R}^{C_c \times H \times W}$ be the feature map extracted by a lightweight context encoder from bathymetry and distance-to-shore.

1. **Projection to Shared Latent Dimension $d$:**
   $$\mathbf{Q} = \mathbf{W}_Q \mathbf{F}_c, \quad \mathbf{K} = \mathbf{W}_K \mathbf{F}_r, \quad \mathbf{V} = \mathbf{W}_V \mathbf{F}_r$$
   where $\mathbf{W}_Q \in \mathbb{R}^{d \times C_c}$, $\mathbf{W}_K, \mathbf{W}_V \in \mathbb{R}^{d \times C}$.
   *Conceptual Meaning:* Projects heterogenous radar features ($C$ channels) and geophysical context features ($C_c$ channels) into a common latent vector space of dimension $d$ (e.g., $d=64$) so that spatial dot-product correlations can be calculated.
2. **Spatial Attention Map Computation:**
   $$\mathbf{A}(i, j) = \text{Softmax}\left(\frac{\mathbf{Q}(i) \mathbf{K}(j)^T}{\sqrt{d}}\right)$$
   The context feature (querying navigability and proximity to shore) computes cross-attention weights over radar features. In non-navigable zones (e.g., cliffs, shallow coral reefs), the attention weight for false bright spots is suppressed towards zero.
3. **Gated Residual Modulation:**
   $$\mathbf{F}_{\text{att}} = \mathbf{A} \mathbf{V}, \quad \mathbf{F}_{\text{out}} = \mathbf{F}_r + \gamma \cdot \text{Conv}_{1\times 1}(\mathbf{F}_{\text{att}})$$
   where $\gamma \in \mathbb{R}$ is a learnable gating parameter initialized to 0.
   *Conceptual Meaning:* The attention matrix filters the radar values ($\mathbf{A} \mathbf{V}$). A learnable residual scalar gate $\gamma$ allows the network to adaptively determine how much context information to blend back into the primary radar features $\mathbf{F}_r$, preserving fine target details while zeroing coastal false alarms.

### 4.3 Prediction Head: Unified 2-Channel CenterNet Architecture
From the modulated feature representation $\mathbf{F}_{\text{out}} \in \mathbb{R}^{C \times H \times W}$, a lightweight convolutional decoder upsamples the features to full spatial resolution $(H, W) = (256, 256)$ and projects directly into a **single 2-channel heatmap tensor**:

$$\hat{\mathbf{Y}} \in [0, 1]^{B \times 2 \times 256 \times 256}$$
* **Channel 0 ($\hat{\mathbf{Y}}_{\text{vessel}}$):** Centroid Gaussian peaks for mobile vessels ($y=1$).
* **Channel 1 ($\hat{\mathbf{Y}}_{\text{structure}}$):** Centroid Gaussian peaks for fixed maritime structures ($y=0$, oil platforms, offshore turbines, buoys).
* **Background ocean pixels:** Decay exponentially to $0.0$.

#### Loss Function: Multi-Channel Modified Focal Loss
The network is trained with a single, unified CenterNet focal loss across both semantic channels:
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}} = -\frac{1}{N_{\text{targets}}} \sum_{c \in \{0, 1\}} \sum_{x, y} \begin{cases} 
(1 - \hat{Y}_{c, xy})^\alpha \log(\hat{Y}_{c, xy}) & \text{if } Y_{c, xy} = 1 \\
(1 - Y_{c, xy})^\beta (\hat{Y}_{c, xy})^\alpha \log(1 - \hat{Y}_{c, xy}) & \text{otherwise}
\end{cases}$$
*Conceptual Meaning:* Eliminates class imbalance between millions of ocean pixels and rare ship pixels. The focal parameter $\alpha=2$ downweights easy background pixels by up to $10.000\times$, while $\beta=4$ softens penalties for predictions slightly off-center ($Y_{c, xy} > 0$).
*Architectural Advantage:* Unifying detection and classification into a single 2-channel heatmap completely eliminates multi-task gradient conflict (negative transfer), removes all loss-weight balancing hyperparameters, and seamlessly detects arbitrary numbers of vessels per patch.

---

## 5. Three-Member Collaborative Division of Labor

```
+---------------------------------------------------------------------------------------+
|                                TEAM DIVISION OF LABOR                                  |
+--------------------------+----------------------------+-------------------------------+
| Member 1: Data & Physics | Member 2: Model & Novelty  | Member 3: Train & Evaluation  |
+--------------------------+----------------------------+-------------------------------+
| Primary: data.py         | Primary: network.py        | Primary: train.py, eval, utils|
| * Tiling / Chipping      | * Backbone Extractor       | * Multi-Task Loss Formulation |
| * Balanced Sampling      | * CGCAM Novelty Module     | * Training & Validation Loop  |
| * Float16 dB Calibration | * Detection / Cls Heads    | * Metric Evaluation Suite     |
| * Dataset & DataLoader   | * Length Regression Head   | * Ablation Study Benchmarking |
| * Scene-Level Split      | * Ablation Toggle Flag     | * Error Analysis & Plots      |
| Slides 1-4 (~3.0 min)    | Slides 5-7 (~3.5 min)      | Slides 8-10 (~3.5 min)        |
+--------------------------+----------------------------+-------------------------------+
```

### 5.1 Member 1: Data & Geophysics Engineer
* **Primary Module:** [`data.py`](file:///Users/lab/Documents/Università/Magistrale/ComputerVision/DarkVesselDetectionCV/data.py) (and constants in [`globals.py`](file:///Users/lab/Documents/Università/Magistrale/ComputerVision/DarkVesselDetectionCV/globals.py)).
* **Core Responsibilities:**
  1. Construct the tiling pipeline to extract $256 \times 256$ chips from scenes via `numpy.memmap`.
  2. Implement balanced patch generation (target-centered positive patches + hard-negative coastal/sea patches).
  3. Encode the radiometric conversion functions (clipping $\sigma^0$ to $[-35, +10]\text{ dB}$ and normalizing bathymetry/shore distance).
  4. Create the PyTorch `Dataset` and multi-worker `DataLoader` with SAR-safe geometric augmentations (random $90^\circ$ rotations, horizontal and vertical flips).
  5. Enforce strict scene-level partitioning across training, validation, and test scenes.
* **Presentation Scope (Slides 1–4, ~3.0 min):**
  - Problem Statement: IUU fishing and the dark vessel surveillance challenge.
  - SAR Fundamentals: Active microwave sensing, $VV$ vs. $VH$ polarimetry, and speckle noise.
  - Data Pipeline: xView3 dataset characteristics, spatial leakage avoidance, and balanced patching.

### 5.2 Member 2: Model & Novelty Architect
* **Primary Module:** [`network.py`](file:///Users/lab/Documents/Università/Magistrale/ComputerVision/DarkVesselDetectionCV/network.py).
* **Core Responsibilities:**
  1. Build the multi-scale feature backbone (e.g., ResNet-18 or ConvNeXt-Tiny adapted for 4-channel input).
  2. Implement the **CGCAM (Context-Guided Cross-Attention Module)** novelty:
     - Context feature projection ($Q$).
     - Radar feature projection ($K, V$).
     - Cross-attention spatial weighting and gated residual connection.
     - Include a boolean toggle parameter `use_cgcam=True/False` for seamless ablation testing.
  3. Implement the unified **2-Channel CenterNet prediction head** outputting heatmap `(Batch_Size, 2, 256, 256)`:
     - Channel 0: Mobile Vessel Centroid peaks.
     - Channel 1: Fixed Offshore Structure Centroid peaks.
* **Presentation Scope (Slides 5–7, ~3.5 min):**
  - State of the Art: Failure modes of standard CNN detectors on radar imagery.
  - Proposed Architecture: System overview and multimodal feature fusion.
  - Novelty Deep-Dive: Mathematical definition of CGCAM and physical justification for context-guided attention.

### 5.3 Member 3: Training, Evaluation & Ablation Specialist
* **Primary Modules:** [`train.py`](file:///Users/lab/Documents/Università/Magistrale/ComputerVision/DarkVesselDetectionCV/train.py), [`evaluation.py`](file:///Users/lab/Documents/Università/Magistrale/ComputerVision/DarkVesselDetectionCV/evaluation.py), [`utils.py`](file:///Users/lab/Documents/Università/Magistrale/ComputerVision/DarkVesselDetectionCV/utils.py).
* **Core Responsibilities:**
  1. Implement the **Multi-Channel Modified Focal Loss** and peak extraction via $3 \times 3$ local max-pooling (NMS).
  2. Construct the training loop with AdamW optimizer, Cosine Annealing learning rate schedule, model checkpointing, and validation logging.
  3. Write evaluation metrics in `utils.py`: Vessel Centroid Precision, Recall, $F_1$-score across varying confidence thresholds, and Offshore Structure discrimination accuracy.
  4. Execute the formal **Ablation Study** (Baseline SAR-only vs. Proposed Model with CGCAM) on the held-out test scenes.
  5. Generate quantitative comparison tables, PR curves, and visual inspection maps highlighting false alarm suppression.
* **Presentation Scope (Slides 8–10, ~3.5 min):**
  - Experimental Setup: Hyperparameters, focal loss convergence curves.
  - Quantitative Results & Ablation Study: $F_1$-score improvements and coastal false alarm suppression.
  - Error Analysis & Conclusions: Diagnostic visual cases (eliminated coastal false alarms) and future research directions.

### 5.4 Parallel Decoupling via Mock Tensors & Git Workflow
To guarantee immediate, unblocked progress:
* **The Shared Contract:** Once [`globals.py`](file:///Users/lab/Documents/Università/Magistrale/ComputerVision/DarkVesselDetectionCV/globals.py) defines the tensor shapes, no team member waits on another.
  * Member 2 develops and verifies [`network.py`](file:///Users/lab/Documents/Università/Magistrale/ComputerVision/DarkVesselDetectionCV/network.py) using mock tensors:
    ```python
    dummy_input = torch.randn(4, 4, 256, 256)
    heatmap_pred = model(dummy_input)  # Shape: (4, 2, 256, 256)
    ```
  * Member 3 writes [`train.py`](file:///Users/lab/Documents/Università/Magistrale/ComputerVision/DarkVesselDetectionCV/train.py) and loss functions using synthetic target heatmaps.
  * Member 1 develops [`data.py`](file:///Users/lab/Documents/Università/Magistrale/ComputerVision/DarkVesselDetectionCV/data.py) to output the exact tensor contract.
* **Git Branching Strategy:**
  - `main`: Protected production branch.
  - `feature/data-pipeline`: Member 1.
  - `feature/cgcam-network`: Member 2.
  - `feature/train-and-ablation`: Member 3.

---

## 6. Contract Interface Specification (`globals.py`)

File: [`globals.py`](file:///Users/lab/Documents/Università/Magistrale/ComputerVision/DarkVesselDetectionCV/globals.py)

```python
"""
globals.py

Global configurations, hyperparameters, device mapping, deterministic random seed,
and contract tensor definitions for Dark Vessel Detection (CGCAM).
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

# Scene-level partitioning to strictly avoid spatial autocorrelation
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
```

---

## 7. Ablation Study & Experimental Protocol

The ablation study is the central criterion for grade excellence in the course. The following two experiments must be trained and compared:

### Benchmark Table Layout for `evaluation.py` and Slides:

| Model Variant | Input Modalities | Novelty Module | Vessel $F_1$ | Precision | Recall | Structure $F_1$ | False Positives / $100\text{ km}^2$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A (Baseline)** | SAR Only ($VV, VH$) | None | *Evaluated* | *Evaluated* | *Evaluated* | *Evaluated* | High (Coastline) |
| **Model B (Early Fusion)** | SAR + Context ($VV, VH, \text{Bathy}, \text{Shore}$) | None (Concat) | *Evaluated* | *Evaluated* | *Evaluated* | *Evaluated* | Medium |
| **Model C (Proposed)** | SAR + Context | **CGCAM (Cross-Attention)** | **Highest** | **Highest** | **Balanced** | **Lowest** | **Near Zero** |

### Evaluation Metrics Protocol:
1. **Centroid Matching:** A prediction is considered a True Positive (TP) if its detected centroid peak is within a Euclidean distance threshold of $R \le 200\text{ meters}$ ($\approx 20\text{ pixels}$) from the ground truth coordinates.
2. **False Positive Suppression Analysis:** Explicitly quantify the false alarm rate in the coastal buffer zone ($d \le 5\text{ km}$ from shore) to demonstrate the physical impact of the CGCAM gating mechanism.

---

## 8. Final Presentation Breakdown (10 Minutes / 10 Slides)

Strictly following the recommended 10-slide outline from the course regulations:

| Slide # | Title / Topic | Speaker | Key Content & Visual Elements |
| :---: | :--- | :---: | :--- |
| **1** | Title & Introduction | **Member 1** | Project title, group members, course info, ALCOR Lab logo. |
| **2** | Outline | **Member 1** | 10-minute presentation road-map. |
| **3** | Problem Statement | **Member 1** | IUU illegal fishing, dark vessels disabling AIS transponders, economic and ecological threats. |
| **4** | SAR Fundamentals & Challenges | **Member 1** | Sentinel-1 physics, $VV$ vs. $VH$ polarimetry, Bragg scattering, multiplicative speckle noise model. |
| **5** | State of the Art | **Member 2** | Traditional detectors (YOLO, Faster R-CNN) and why early concatenation fails near coastlines. |
| **6** | Proposed Method & Architecture | **Member 2** | Full system block diagram: ResNet backbone, Context Encoder, and Multi-Task heads. |
| **7** | CGCAM Novelty Deep-Dive | **Member 2** | Mathematical formulation of Context-Guided Cross-Attention; physical explanation of coastal gating. |
| **8** | Dataset & Experimental Setup | **Member 3** | xView3 scene-level split, balanced patch generation, multi-task loss balancing ($\mathcal{L}_{\text{det}}, \mathcal{L}_{\text{cls}}, \mathcal{L}_{\text{len}}$). |
| **9** | Model Evaluation & Ablation Study | **Member 3** | Quantitative comparison table (Baseline vs. CGCAM), $F_1$-score gains, visual evidence of eliminated false alarms. |
| **10** | Conclusions & Future Work | **Member 3** | Summary of engineering achievements, deployment feasibility on embedded edge devices, references. |

---

> [!NOTE]
> **Companion Mathematical & Physical Formulary:**  
> For the exhaustive definition of every physical variable, mathematical domain, index, and conceptual breakdown across all equations used throughout this project, please consult the companion handbook:  
> 📄 [FORMULARY.md](./FORMULARY.md)

---
*Document generated for the Computer Vision Master's examination repository at Sapienza University of Rome.*

