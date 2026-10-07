# Dark Vessel Detection via Multimodal SAR & Context-Guided Cross-Attention (CGCAM)

**Master's Course in Computer Vision (A.Y. 2025–2026)** \
**Department of Computer, Control and Management Engineering (DIAG)**\
**Sapienza University of Rome**\
**Instructor:** Prof. Irene Amerini

---

## 📌 Project Overview

End-to-end Deep Learning framework for **Dark Vessel Detection** in Synthetic Aperture Radar (SAR) imagery, built on the **xView3-SAR** benchmark (Copernicus Sentinel-1).

The task: vessels engaged in IUU (illegal, unreported, unregulated) fishing, smuggling and sanctions evasion switch off their AIS transponders and become *"dark"*. Optical satellites cannot see them at night or under cloud cover; SAR can. The difficulty is that radar imagery is dominated by two false-alarm families — **wind-driven wave crests** in open water (sea-state clutter) and **layover / foreshortening** of coastal terrain (geometric distortion).

The primary contribution is **CGCAM (Context-Guided Cross-Attention Module)**: a multimodal attention block that keeps the radar stream (VV, VH) and the geophysical context stream (bathymetry, distance-to-shore, wind speed) **decoupled** until a learned cross-attention gate, so that geophysical priors can suppress false alarms without being mixed into the radar features by early convolution.

### Architecture Highlights

* **Input:** 5 normalized channels — `VH_dB`, `VV_dB`, `bathymetry`, `distance_to_shore`, `wind_speed`.
* **Radar Backbone:** multi-scale residual CNN on Sentinel-1 dual-polarization (VV, VH).
* **Geophysical Context Encoder:** lightweight CNN on bathymetric depth, distance-to-shore (derived, see §2), and OWI wind speed.
* **CGCAM (novelty):** scaled dot-product cross-attention where context features act as **queries (Q)** and radar features as **keys/values (K, V)**, blended back through a learnable residual gate `gamma`. The intended behavior is CFAR-like: radar returns are trusted only where they exceed the wind-predicted Bragg clutter background and the geophysical context declares the pixel navigable.
* **Prediction head:** unified 2-channel anchor-free **CenterNet** heatmap — channel 0: mobile vessel centroids; channel 1: fixed offshore structures (platforms, buoys).
* **Objective:** a single multi-channel modified focal loss (no auxiliary heads, no loss-balancing hyperparameters).

### Four Ablation Variants

All four share the **same tensor identity** `(B, 5, 256, 256) -> (B, 2, 256, 256)` and the same training pipeline; only the routing of the context channels differs (`FUSION_MODE` in `globals.py`):

| `FUSION_MODE` | Variant | Routing |
|---|---|---|
| `sar_only` | **A** (baseline) | only VV/VH reach the radar backbone (context zeroed) |
| `early_fusion` | **B** (baseline) | all 5 channels enter the backbone's first convolution |
| `cgcam_no_wind` | **C−wind** | decoupled streams, context wind channel zeroed |
| `cgcam` | **C** (proposed) | radar `0:2` and context `2:5` stay separate until CGCAM |

The two CGCAM variants have identical parameter shapes and are trained under identical optimization settings, so C vs C−wind is a controlled comparison of the wind channel alone.

---

## 📁 Repository Structure

```
DarkVesselDetectionCV/
├── globals.py               # Contract constants, device config, scene splits, seeds
├── data.py                  # GeoTIFF loading, radiometric/context normalization, derived
│                            #   distance-to-shore (EDT), Gaussian heatmaps, DataLoader
├── network.py               # DarkVesselNet: radar backbone + context encoder + CGCAM + CenterNet head
├── optimizers.py            # Optimizer factory (AdamW/Adam/SGD/RMSprop/Lion) + search spaces
├── train.py                 # Train/validate passes, best-checkpoint selection
├── evaluation.py            # Peak extraction + centroid matching, PR/loss curves,
│                            #   near-shore FP analysis, ablation runner
├── tuning_runtime.py        # Seeded loaders, synthetic smoke data, detection caching,
│                            #   code/dataset fingerprints (reproducibility)
├── run_ablation_study.py    # Final 4-variant ablation over seeds (entry point)
├── PRESENTATION.pdf         # Project presentation
├── LICENSE                  # GNU GPL v3
└── README.md
```

---

## 🚀 Getting Started

### 1. Requirements

* Python >= 3.11
* PyTorch >= 2.0
* NumPy, SciPy, Pandas, tifffile, OpenCV (`cv2`), Matplotlib
* Optuna (for the hyperparameter search used by the tuning pipeline)

### 2. Dataset Setup

The project uses the open **xView3-SAR** benchmark (Sentinel-1 scenes with vessel/structure annotations):
**[https://iuu.xview.us/](https://iuu.xview.us/)**

Place the extracted scenes and annotations under `dataset/` (gitignored — multi-GB rasters):

```
DarkVesselDetectionCV/
└── dataset/
    ├── labels.csv                  # xView3 16-column annotation file
    ├── 05bc615a9b0e1159t/          # one folder per scene
    │   ├── VH_dB.tif, VV_dB.tif    # SAR backscatter (float16, ~22k x 30k px)
    │   ├── bathymetry.tif          # GEBCO ocean depth
    │   ├── owiMask.tif             # land / ocean mask
    │   ├── owiWindSpeed.tif        # OWI wind speed (m/s)
    │   ├── owiWindDirection.tif, owiWindQuality.tif
    └── ...
```

Two data facts worth knowing before you run anything:

* **`distance_to_shore.tif` does not exist in the release.** xView3 provides only `distance_from_shore_km` as a scalar per detection. `data.py` derives the dense field from `owiMask.tif` with a Euclidean distance transform (EDT) and validates it against the label scalars (measured: MAE 1.148 km, Pearson 0.884 on 974 labelled detections; a scene-dependent ~1 km translational offset vs the labels' high-resolution shoreline remains and is documented).
* **OWI wind nodata is encoded as 0.0** (`normalize_wind_speed`), the same value as a measured 0 m/s. Only ~27% of OWI wind pixels are valid, and the gap is ~2x denser within 5 km of shore. Missing wind therefore degrades to "expect low clutter" near the coast. Known limitation — see below.

### 3. Verify the Architecture Contract

```bash
python network.py
```

Real output (ranges vary with random init):

```text
Testing DarkVesselNet architecture contract...
A / SAR only: (4, 5, 256, 256) -> (4, 2, 256, 256), range=[0.1006, 0.1007]
B / early fusion: (4, 5, 256, 256) -> (4, 2, 256, 256), range=[0.1006, 0.1007]
C-wind / CGCAM without wind: (4, 5, 256, 256) -> (4, 2, 256, 256), range=[0.1006, 0.1007]
C / CGCAM: (4, 5, 256, 256) -> (4, 2, 256, 256), range=[0.1006, 0.1007]
Architecture contract verified successfully!
```

`python data.py` runs the analogous pipeline smoke test (batch shapes `(4, 5, 256, 256)` / `(4, 2, 256, 256)`, per-channel value ranges).

### 4. Run the Ablation Study

`run_ablation_study.py` is the entry point. For each fusion variant it loads the best optimizer/hyperparameters found by the Optuna search (`tuning_runs/optuna.db`), retrains with fresh seeds, **selects the checkpoint epoch and the confidence threshold on the validation split**, scores the test split **once**, and reports mean ± std over seeds (vessel and structure channels, PR curves, prediction overlays).

```bash
QUICK=1 python run_ablation_study.py   # 1 seed, 2 epochs: pipeline check end-to-end
python run_ablation_study.py           # full run (3 seeds x 30 epochs x 4 variants)
```

Outputs land in `results/` (gitignored): `ablation_aggregate.json` (mean/std + configs + seeds), `ablation_table.json`, PR/loss curves, detection overlays.

---

## ⚙️ Configuration & Contract

All constants live in `globals.py` (single source of truth, imported by every module).

| Constant | Value | Meaning |
|---|---|---|
| `INPUT_CHANNELS` | 5 | VH, VV, bathymetry, distance_to_shore, wind_speed |
| `NUM_CLASSES` | 2 | ch0 mobile vessel, ch1 fixed offshore structure |
| `PATCH_SIZE` | 256 | patch side (px) |
| `PATCH_JITTER_PX` | 108 | train-only jitter of positive patch origin; target centroid lands in [20, 236] so the Gaussian target (radius 3σ = 6 px) is never truncated by patch borders |
| `GAUSSIAN_RADIAL_SIGMA` | 2.0 | Gaussian heatmap target radius (px) |
| `FOCAL_ALPHA` / `FOCAL_BETA` | 2.0 / 4.0 | modified focal loss exponents |
| `MATCH_DISTANCE_PIXELS` | 20 | centroid matching radius = 200 m at 10 m/px |
| `PEAK_CONFIDENCE_THRESHOLD` | 0.3 | default detection threshold (ablation selects its own on validation) |
| `NMS_KERNEL_SIZE` | 3 | 3x3 max-pooling peak extraction |
| `MAX_WIND_SPEED_MS` | 25.0 | wind clip/scale to [0, 1] |
| `MAX_DISTANCE_SHORE_METERS` | 50000 | distance-to-shore clip/scale to [0, 1] |
| `MAX_BATHYMETRY_METERS` | -2000 | bathymetry clip/scale to [0, 1] |
| `VH_MIN/MAX_DB`, `VV_MIN/MAX_DB` | −35/+10, −30/+15 | per-channel radiometric ranges |

**Scene-level split** (deliberately not a random patch split — patches from one swath share sensor geometry, wind and clutter, so a random split leaks; the model is evaluated on unseen geographic swaths):

| Split | Scenes |
|---|---|
| Train | `05bc615a9b0e1159t`, `2899cfb18883251bt`, `72dba3e82f782f67t`, `cbe4ad26fe73f118t` |
| Val | `e98ca5aba8849b06t` |
| Test | `590dd08f71056cacv`, `b1844cde847a3942v` |

---

## 📊 Results

6 seeds (two independent batches of 3), 30 epochs, mean ± std. Per-variant optimizer/hyperparameters come from the Optuna search; the two CGCAM variants share one identical config by construction. Confidence threshold and checkpoint epoch selected **on validation**, test scored once.

| Variant | Precision | Recall | F1 | FP | near-shore FP | near-shore FP share |
|---|---|---|---|---|---|---|
| A `sar_only` | 0.626 ± 0.013 | **0.603 ± 0.069** | **0.613 ± 0.037** | 2243 ± 251 | 760 ± 137 | 0.34 |
| B `early_fusion` | 0.579 ± 0.030 | 0.347 ± 0.091 | 0.430 ± 0.074 | 1561 ± 293 | 483 ± 86 | 0.31 |
| C−wind `cgcam_no_wind` | **0.690 ± 0.040** | 0.465 ± 0.111 | 0.550 ± 0.092 | **1286 ± 267** | **467 ± 121** | 0.36 |
| C `cgcam` | 0.689 ± 0.025 | 0.474 ± 0.075 | 0.559 ± 0.052 | 1343 ± 281 | 505 ± 127 | 0.37 |

What these numbers support, honestly stated:

* **CGCAM is more precise than the SAR-only baseline** (0.689 vs 0.626, Δ = +0.063, z = 2.2 — the one comparison in the study clearing the 2σ bar; same sign in both seed batches). The gain is *selectivity*, not just fewer detections: false positives fall to x0.60 of the baseline while true positives fall only to x0.79 (FP/TP ratio 0.756 vs 1.0 expected under a pure volume effect). `early_fusion` is the control that makes the test meaningful — its FP/TP ratio is 1.221, i.e. its lower FP count is entirely a volume artifact.
* **The baseline keeps the higher recall and F1** (0.603 / 0.613). CGCAM buys precision at a recall cost; it does not dominate every metric.
* **The wind channel shows no measurable effect.** C vs C−wind differ by less than z = 0.25 on every quantity (precision Δ −0.001, near-shore FP Δ +38), and the offshore false-positive delta flips sign between the two independent seed batches (−69 vs +109). Any hypothesis that the wind channel selectively removes offshore wave-clutter false alarms is withdrawn on this evidence.

**Known limitations** (all documented, none hidden):

1. **`near_shore_fp_rate` is a share of each model's own false positives, not a density per area** — read it as "what fraction of FPs lie within 5 km of shore", not "FP per 100 km²".
2. **Near-shore *recall* is not measured.** "Fewer coastal false alarms" must not be read as "better coastal performance": without the coastal TP/FN split, we cannot rule out that coastal targets are lost alongside them.
3. **The OWI wind product covers only ~27% of valid pixels**, with the gap ~2x denser within 5 km of shore, and missing values share the encoding of a measured calm (0.0). This is a real data defect worth fixing on its own merits (a validity mask or gap interpolation), but it is **not** presented as the explanation of any observed result.
4. **Sample size: 7 scenes.** Proof-of-concept scale; per-scene spread is wide. Per-scene metrics should accompany any aggregate claim.
5. **Derived distance-to-shore carries a scene-dependent ~1 km offset** vs the labels' high-resolution shoreline (see §2); a finer coastline reconstruction was considered and is *not* recommended without an ablation, since SAR-derived coastlines are corrupted exactly in the coastal strip where accuracy matters.

---

## 📚 References

1. **Paolo, F., et al.** (2022). *xView3-SAR: Detecting Dark Fishing Activity Using Synthetic Aperture Radar Imagery.* NeurIPS Datasets & Benchmarks. arXiv:2206.00897 — dataset and benchmark.
2. **Zhou, X., Wang, D., & Krähenbühl, P.** (2019). *Objects as Points.* arXiv:1904.07850 — CenterNet anchor-free detection (heatmap formulation).
3. **Lin, T.-Y., et al.** (2017). *Focal Loss for Dense Object Detection.* ICCV 2017 — the loss this project's objective is derived from.
4. **Chen, X., et al.** (2023). *Symbolic Discovery of Optimization Algorithms (Lion).* NeurIPS 2023. arXiv:2302.06675 — `Lion` optimizer in `optimizers.py` (small independent implementation).
5. **Bai, Y., et al.** (2024). *FAD-SAR: Fishing Activity Detection via SAR and Deep Learning.* arXiv:2404.18245 — realistic baseline anchor on xView3 (Avg-F1 ≈ 0.216 at fixed operating points).
6. **Laganier, C., et al.** (2025). *Efficient SAR Vessel Detection for FPGA-Based On-Satellite Sensing.* ACM/IEEE SEC '25. arXiv:2507.04842 — edge/on-satellite deployment context.

---

## 📄 License

GNU General Public License v3.0 — see [`LICENSE`](./LICENSE).
