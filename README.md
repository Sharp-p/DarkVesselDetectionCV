# Dark Vessel Detection via Multimodal SAR & Context-Guided Cross-Attention (CGCAM)

**Master's Course in Computer Vision (A.Y. 2025–2026)**  
**ALCOR Lab, Department of Computer, Control and Management Engineering (DIAG)**  
**Sapienza University of Rome**  
**Instructor:** Prof. Irene Amerini  

---

## 📌 Project Overview
This repository implements an end-to-end Deep Learning framework for **Dark Vessel Detection** in Synthetic Aperture Radar (SAR) imagery using the **xView3-SAR** benchmark (Copernicus Sentinel-1). 

The primary contribution is **CGCAM (Context-Guided Cross-Attention Module)**, a novel multimodal attention architecture that decouples active radar backscatter ($VV, VH$) from auxiliary geophysical context (bathymetry and distance-to-shore) to suppress severe coastal false alarms (*layover* and *foreshortening*).

### Architecture Highlights
* **Backbone:** Multi-scale residual CNN processing Sentinel-1 dual-polarization $VV$ and $VH$ imagery.
* **Geophysical Context Encoder:** Lightweight CNN processing bathymetric depth and distance-to-shore maps.
* **CGCAM (Novelty):** Scaled dot-product cross-attention where geographic navigation queries ($Q$) modulate radar keys and values ($K, V$) through a learnable residual gate ($\gamma$).
* **Prediction Head:** Unified 2-Channel Anchor-Free **CenterNet** spatial heatmap:
  * **Channel 0:** Mobile Vessel Centroids ($\hat{\mathbf{Y}}_{\text{vessel}}$).
  * **Channel 1:** Fixed Offshore Structures ($\hat{\mathbf{Y}}_{\text{structure}}$, oil rigs, offshore turbines, buoys).
* **Loss Function:** Multi-Channel Penalty-Reduced Modified Focal Loss.

---

## 📁 Repository Structure
```
DarkVesselDetectionCV/
├── globals.py               # Constants, device config, scene splits, and tensor contracts
├── network.py               # DarkVesselNet (Radar Backbone + Context Encoder + CGCAM + CenterNet Head)
├── data.py                  # Tiling, balanced patch sampling, and PyTorch DataLoader
├── train.py                 # Multi-channel focal loss training loop and optimizer setup
├── evaluation.py            # Scene-level evaluation, PR curves, and ablation study benchmarks
├── utils.py                 # 3x3 max-pooling peak extraction, spatial matching, and F1 metrics
└── README.md
```

---

## 🚀 Getting Started

### 1. Requirements
* Python $\ge 3.10$
* PyTorch $\ge 2.0$
* NumPy, OpenCV, Matplotlib

### 2. Verify Architecture Contract
```bash
python network.py
```
Expected output:
```text
Model Input shape:  torch.Size([2, 4, 256, 256])
Model Output shape: torch.Size([2, 2, 256, 256])
Architecture contract verified successfully!
```