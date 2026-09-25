# Complete Mathematical & Physical Formulary
## Dark Vessel Detection via Multimodal SAR & Context-Guided Cross-Attention (CGCAM)

**Course:** Computer Vision (A.Y. 2025–2026) — Sapienza University of Rome  
**Associated Document:** [`PROJECT_PLAN.md`](./PROJECT_PLAN.md)  
**Target Audience:** Computer Science & Artificial Intelligence Students  

---

## Table of Contents
1. [Radar & SAR Physical Geometry](#1-radar--sar-physical-geometry)
   - [1.1 Radar Range Equation (Slant Range)](#11-radar-range-equation-slant-range)
   - [1.2 Real Aperture Radar (RAR) Azimuth Resolution](#12-real-aperture-radar-rar-azimuth-resolution)
   - [1.3 Synthetic Aperture Radar (SAR) Azimuth Resolution](#13-synthetic-aperture-radar-sar-azimuth-resolution)
   - [1.4 Slant Range Resolution](#14-slant-range-resolution)
   - [1.5 Ground Range Resolution](#15-ground-range-resolution)
2. [Signal Representation & Radiometry](#2-signal-representation--radiometry)
   - [2.1 Complex SAR Signal (Single Look Complex - SLC)](#21-complex-sar-signal-single-look-complex---slc)
   - [2.2 Normalized Radar Cross Section (Sigma Nought, $\sigma^0$)](#22-normalized-radar-cross-section-sigma-nought-sigma0)
   - [2.3 Decibel (dB) Conversion](#23-decibel-db-conversion)
   - [2.4 Multiplicative Speckle Noise Model](#24-multiplicative-speckle-noise-model)
   - [2.5 Bragg Scattering Resonance Wavelength](#25-bragg-scattering-resonance-wavelength)
   - [2.6 Target-to-Clutter Ratio (TCR)](#26-target-to-clutter-ratio-tcr)
3. [Geospatial & Radiometric Normalization](#3-geospatial--radiometric-normalization)
   - [3.1 Radiometric Min-Max Standardization (dB Scale to [0, 1])](#31-radiometric-min-max-standardization-db-scale-to-0-1)
   - [3.2 Distance-to-Shore Normalization](#32-distance-to-shore-normalization)
   - [3.3 Ocean Bathymetry Normalization](#33-ocean-bathymetry-normalization)
4. [Deep Learning Architecture & Novelty (CGCAM)](#4-deep-learning-architecture--novelty-cgcam)
   - [4.1 Linear Projections to Latent Space $d$](#41-linear-projections-to-latent-space-d)
   - [4.2 Scaled Dot-Product Spatial Cross-Attention](#42-scaled-dot-product-spatial-cross-attention)
   - [4.3 Gated Residual Modulation](#43-gated-residual-modulation)
5. [Multi-Task Objective Functions & Optimization](#5-multi-task-objective-functions--optimization)
   - [5.1 Modified Focal Loss for Centroid Detection](#51-modified-focal-loss-for-centroid-detection)
   - [5.2 Binary Cross-Entropy Loss for Vessel Classification](#52-binary-cross-entropy-loss-for-vessel-classification)
   - [5.3 Smooth L1 Loss for Hull Length Regression](#53-smooth-l1-loss-for-hull-length-regression)
   - [5.4 Total Multi-Task Loss](#54-total-multi-task-loss)
6. [Academic Grading Formulation](#6-academic-grading-formulation)
   - [6.1 Final Course Grade Composition](#61-final-course-grade-composition)

---

## 1. Radar & SAR Physical Geometry

```
               Satellite (Orbiting at altitude H, speed v)
                     |\
                     | \
                     |  \  Slant Range R (line-of-sight distance)
               Nadir |   \
                     |    \  Incidence Angle (theta_i)
                     |     \
=====================+======*=============================== Earth / Sea Surface
                   Nadir   Target (Ground Range distance)
```

### 1.1 Radar Range Equation (Slant Range)
$$R = \frac{c \cdot \Delta t}{2}$$

* **Conceptual Meaning:** Calculates the straight-line Euclidean distance (*Slant Range*) between the satellite antenna and a target on Earth by measuring the round-trip flight time of a microwave pulse.
* **Terms & Parameters:**
  | Symbol | Name | Domain | Definition & Physical Unit |
  | :--- | :--- | :--- | :--- |
  | $R$ | Slant Range Distance | $\mathbb{R}^+$ | The line-of-sight distance from satellite to target (meters $[\text{m}]$). |
  | $c$ | Speed of Light | Constant | Universal speed of electromagnetic wave propagation in vacuum/air ($\approx 3 \cdot 10^8\ \text{m/s}$). |
  | $\Delta t$ | Two-Way Travel Time | $\mathbb{R}^+$ | Time elapsed between pulse transmission and echo reception (seconds $[\text{s}]$). |
  | $2$ | Round-Trip Path Factor | Constant | Compensates for the fact that the wave travels to the target and back. |

---

### 1.2 Real Aperture Radar (RAR) Azimuth Resolution
$$\delta_{az, \text{RAR}} = R \cdot \theta_{az} \approx \frac{\lambda R}{L}$$

* **Conceptual Meaning:** In a conventional stationary antenna (*Real Aperture Radar*), optical diffraction spreads the microwave beam as it travels. Consequently, spatial resolution along the flight track (*Azimuth*) degrades linearly with distance.
* **Terms & Parameters:**
  | Symbol | Name | Domain | Definition & Physical Unit |
  | :--- | :--- | :--- | :--- |
  | $\delta_{az, \text{RAR}}$ | RAR Azimuth Resolution | $\mathbb{R}^+$ | Minimum distance along the flight path between two distinguishable targets ($[\text{m}]$). |
  | $R$ | Slant Range Distance | $\mathbb{R}^+$ | Range from radar to ground target ($[\text{m}]$). |
  | $\theta_{az}$ | Azimuth Beamwidth | $\mathbb{R}^+$ | Angular angular spread of the radiated microwave beam (radians $[\text{rad}]$). |
  | $\lambda$ | Radar Wavelength | $\mathbb{R}^+$ | Carrier wavelength (meters $[\text{m}]$, $\approx 0.0554\text{ m}$ for Sentinel-1 C-band). |
  | $L$ | Antenna Physical Length | $\mathbb{R}^+$ | Physical aperture length of the antenna mounted on the satellite ($[\text{m}]$). |

---

### 1.3 Synthetic Aperture Radar (SAR) Azimuth Resolution
$$\delta_{az, \text{SAR}} = \frac{L}{2}$$

* **Conceptual Meaning:** As the satellite moves, it illuminates a ground target from multiple successive orbital positions, recording both amplitude and Doppler phase. A matched filter reconstructs an equivalent *synthetic antenna* kilometers long. Incredibly, the resulting spatial resolution becomes **independent of orbital altitude $R$ and wavelength $\lambda$**, depending solely on antenna length.
* **Terms & Parameters:**
  | Symbol | Name | Domain | Definition & Physical Unit |
  | :--- | :--- | :--- | :--- |
  | $\delta_{az, \text{SAR}}$ | SAR Azimuth Resolution | $\mathbb{R}^+$ | The theoretical along-track resolution limit ($[\text{m}]$). |
  | $L$ | Physical Antenna Length | $\mathbb{R}^+$ | Real length of the antenna ($[\text{m}]$). For Sentinel-1, $L \approx 12.3\text{ m} \implies \delta_{az} \approx 6.1\text{ m}$. |
  | $2$ | Two-Way Doppler Factor | Constant | Arises from the two-way phase history during coherent synthesis. |

---

### 1.4 Slant Range Resolution
$$\delta_r = \frac{c}{2B}$$

* **Conceptual Meaning:** Defines how well the radar can separate two targets along the oblique line-of-sight direction. Instead of using infinitesimally short pulses (which lack energy), SAR transmits frequency-modulated *chirps* of bandwidth $B$, achieving fine time resolution through pulse compression.
* **Terms & Parameters:**
  | Symbol | Name | Domain | Definition & Physical Unit |
  | :--- | :--- | :--- | :--- |
  | $\delta_r$ | Slant Range Resolution | $\mathbb{R}^+$ | Minimum resolvable distance along the radar beam axis ($[\text{m}]$). |
  | $c$ | Speed of Light | Constant | $\approx 3 \cdot 10^8\ \text{m/s}$. |
  | $B$ | Chirp Bandwidth | $\mathbb{R}^+$ | Frequency excursion bandwidth of the transmitted radar pulse (Hertz $[\text{Hz}]$). |
  | $2$ | Round-Trip Factor | Constant | Accounts for the wave travelling to and from the target. |

---

### 1.5 Ground Range Resolution
$$\delta_{gr} = \frac{\delta_r}{\sin \theta_i} = \frac{c}{2B \sin \theta_i}$$

* **Conceptual Meaning:** The slant range resolution $\delta_r$ is measured obliquely in space. To know the true resolution projected horizontally onto the sea or land surface (*Ground Range*), we project $\delta_r$ using trigonometry through the local incidence angle.
* **Terms & Parameters:**
  | Symbol | Name | Domain | Definition & Physical Unit |
  | :--- | :--- | :--- | :--- |
  | $\delta_{gr}$ | Ground Range Resolution | $\mathbb{R}^+$ | Spatial resolution projected on the horizontal Earth surface ($[\text{m}]$). |
  | $\delta_r$ | Slant Range Resolution | $\mathbb{R}^+$ | Radial resolution along the beam line ($[\text{m}]$). |
  | $\theta_i$ | Local Incidence Angle | $(0, \frac{\pi}{2})$ | Angle between the incoming radar ray and the local surface normal vector. |
  | $\sin \theta_i$ | Projection Factor | $(0, 1)$ | Trigonometric scaling factor projecting slant range onto horizontal ground range. |

---

## 2. Signal Representation & Radiometry

### 2.1 Complex SAR Signal (Single Look Complex - SLC)
$$S(x, y) = I(x, y) + j \cdot Q(x, y) = A(x, y) \cdot e^{j \phi(x, y)}$$

* **Conceptual Meaning:** Before amplitude detection, SAR data is fundamentally complex-valued. Each pixel stores two orthogonal components of the returned electromagnetic wave: amplitude (energy backscattered) and coherent phase (precise millimeter wave travel distance).
* **Terms & Parameters:**
  | Symbol | Name | Domain | Definition & Physical Unit |
  | :--- | :--- | :--- | :--- |
  | $S(x, y)$ | Complex Radar Signal | $\mathbb{C}$ | Complex value at image coordinate pixel $(x, y)$. |
  | $I(x, y)$ | In-Phase Component | $\mathbb{R}$ | Real part of the received signal vector. |
  | $Q(x, y)$ | Quadrature Component | $\mathbb{R}$ | Imaginary part (shifted by $90^\circ$). |
  | $j$ | Imaginary Unit | Constant | $j = \sqrt{-1}$. |
  | $A(x, y)$ | Signal Amplitude | $\mathbb{R}^+$ | Wave magnitude: $A = \sqrt{I^2 + Q^2}$. Proportional to reflected power. |
  | $\phi(x, y)$ | Coherent Phase | $[-\pi, \pi)$ | Phase angle: $\phi = \text{atan2}(Q, I)$. Sensitive to terrain height/displacement. |

---

### 2.2 Normalized Radar Cross Section (Sigma Nought, $\sigma^0$)
$$\sigma^0 = \frac{\langle \sigma \rangle}{A_{\text{ground}}}$$

* **Conceptual Meaning:** Measures the electromagnetic reflectivity of a distributed continuous surface (like the ocean). It normalizes the average radar cross section of all micro-scatterers by the physical ground area of the resolution cell.
* **Terms & Parameters:**
  | Symbol | Name | Domain | Definition & Physical Unit |
  | :--- | :--- | :--- | :--- |
  | $\sigma^0$ | Sigma Nought (Backscatter) | $\mathbb{R}^+$ | Dimensionless normalized backscattering coefficient ($[\text{m}^2 / \text{m}^2]$). |
  | $\langle \sigma \rangle$ | Mean Radar Cross Section | $\mathbb{R}^+$ | Expected radar cross-sectional reflective area of the target ($[\text{m}^2]$). |
  | $A_{\text{ground}}$ | Physical Ground Area | $\mathbb{R}^+$ | Footprint area of a single resolution cell projected onto the ground ($[\text{m}^2]$). |

---

### 2.3 Decibel (dB) Conversion
$$\sigma^0_{[\text{dB}]} = 10 \cdot \log_{10}(\sigma^0_{\text{linear}})$$

* **Conceptual Meaning:** Radar backscatter spans several orders of magnitude (from $0.0001$ for flat calm sea to $>1000$ for metallic superstructures). The decibel scale compresses this exponential distribution into a manageable linear-like scale for analysis and visualization.
* **Terms & Parameters:**
  | Symbol | Name | Domain | Definition & Physical Unit |
  | :--- | :--- | :--- | :--- |
  | $\sigma^0_{[\text{dB}]}$ | Backscatter in Decibels | $\mathbb{R}$ | Logarithmic radiometric intensity ($[\text{dB}]$). Typical range: $-35\text{ dB}$ to $+15\text{ dB}$. |
  | $\sigma^0_{\text{linear}}$ | Linear Power Ratio | $\mathbb{R}^+$ | Raw power ratio ($[\text{m}^2/\text{m}^2]$). |
  | $10 \log_{10}$ | Decibel Operator | Function | Standard acoustic/electromagnetic power scaling transformation. |

---

### 2.4 Multiplicative Speckle Noise Model
$$I(x, y) = R(x, y) \cdot \eta(x, y)$$

* **Conceptual Meaning:** Unlike optical noise which is typically additive Gaussian ($I = R + \mathcal{N}$), SAR speckle is fundamentally **multiplicative**. It is caused by the constructive and destructive interference of coherent waves bouncing off sub-pixel microscopic scatterers. As a result, brighter surfaces exhibit larger absolute noise fluctuations.
* **Terms & Parameters:**
  | Symbol | Name | Domain | Definition & Physical Unit |
  | :--- | :--- | :--- | :--- |
  | $I(x, y)$ | Observed Pixel Intensity | $\mathbb{R}^+$ | Raw corrupted intensity value observed by the detector. |
  | $R(x, y)$ | Ground Truth Reflectivity | $\mathbb{R}^+$ | True underlying radar cross section to be estimated. |
  | $\eta(x, y)$ | Speckle Noise Factor | $\mathbb{R}^+$ | Stationary stochastic noise process distributed as $\eta \sim \Gamma(L, 1/L)$. |
  | $L$ | Number of Looks | $\mathbb{N}^+$ | Number of independent spatial looks averaged during image generation. |
  | $\mathbb{E}[\eta]$ | Expected Noise Mean | Constant | Equal to $1.0$ (unbiased on average). |
  | $\text{Var}[\eta]$ | Noise Variance | $\mathbb{R}^+$ | Equal to $1/L$. Decreases as more independent looks are multilooked. |

---

### 2.5 Bragg Scattering Resonance Wavelength
$$\Lambda_{\text{sea}} = \frac{\lambda}{2 \sin \theta_i}$$

* **Conceptual Meaning:** Explains why rough seas look bright in $VV$ polarization. When wind creates periodic surface ripples whose physical crest-to-crest spacing $\Lambda_{\text{sea}}$ matches this equation, consecutive radar waves reflect in phase, constructively reinforcing the returned echo.
* **Terms & Parameters:**
  | Symbol | Name | Domain | Definition & Physical Unit |
  | :--- | :--- | :--- | :--- |
  | $\Lambda_{\text{sea}}$ | Ocean Wave Ripple Spacing | $\mathbb{R}^+$ | Physical wavelength of the ocean capillary/gravity waves causing resonance ($[\text{m}]$). |
  | $\lambda$ | Radar Carrier Wavelength | $\mathbb{R}^+$ | $\approx 0.0554\text{ m}$ for Sentinel-1 C-band. |
  | $\theta_i$ | Radar Incidence Angle | $(0, \frac{\pi}{2})$ | Angle between the radar beam and the ocean surface normal. |

---

### 2.6 Target-to-Clutter Ratio (TCR)
$$\text{TCR} = \frac{\sigma^0_{\text{target}}}{\sigma^0_{\text{clutter}}}$$

* **Conceptual Meaning:** Evaluates how prominently a ship stands out against the ambient ocean background. In $VV$, sea clutter is high, so TCR is low. In $VH$, sea clutter is nearly zero while ship dihedral reflections remain strong, making $\text{TCR}_{VH}$ immensely higher.
* **Terms & Parameters:**
  | Symbol | Name | Domain | Definition & Physical Unit |
  | :--- | :--- | :--- | :--- |
  | $\text{TCR}$ | Target-to-Clutter Ratio | $\mathbb{R}^+$ | Dimensionless contrast ratio (often expressed in $\text{dB}$ as $\sigma^0_{\text{target, dB}} - \sigma^0_{\text{clutter, dB}}$). |
  | $\sigma^0_{\text{target}}$ | Target Backscatter | $\mathbb{R}^+$ | Backscatter returned by the metallic vessel hull ($[\text{m}^2/\text{m}^2]$). |
  | $\sigma^0_{\text{clutter}}$ | Clutter Backscatter | $\mathbb{R}^+$ | Ambient backscatter returned by the surrounding sea surface ($[\text{m}^2/\text{m}^2]$). |

---

## 3. Geospatial & Radiometric Normalization

### 3.1 Radiometric Min-Max Standardization (dB Scale to [0, 1])
$$\sigma^0_{[\text{norm}]} = \text{clip}\left(\frac{\sigma^0_{[\text{dB}]} - \sigma^0_{\min}}{\sigma^0_{\max} - \sigma^0_{\min}}, 0.0, 1.0\right)$$

* **Conceptual Meaning:** Linearly standardizes raw radar decibel values into the bounded range $[0.0, 1.0]$ required for stable neural network training (preventing gradient explosion or early activation saturation). Values outside the realistic physical range are clamped.
* **Terms & Parameters:**
  | Symbol | Name | Domain | Definition & Physical Unit |
  | :--- | :--- | :--- | :--- |
  | $\sigma^0_{[\text{norm}]}$ | Normalized Radar Pixel | $[0.0, 1.0]$ | Dimensionless input feature value passed into the first convolution layer. |
  | $\sigma^0_{[\text{dB}]}$ | Input Backscatter | $\mathbb{R}$ | Original decibel value stored in the GeoTIFF file. |
  | $\sigma^0_{\min}$ | Minimum Clipping Floor | Constant | Set to $-35.0\text{ dB}$ (any lower value is sensor thermal noise). |
  | $\sigma^0_{\max}$ | Maximum Clipping Ceiling | Constant | Set to $+10.0\text{ dB}$ (strong specular/corner metallic reflection). |
  | $\text{clip}(x, a, b)$ | Clamping Function | Function | $\text{clip}(x, a, b) = \max(a, \min(x, b))$. |

---

### 3.2 Distance-to-Shore Normalization
$$\text{DistShore}_{[\text{norm}]} = \text{clip}\left(\frac{D}{D_{\max}}, 0.0, 1.0\right)$$

* **Conceptual Meaning:** Scales metric distance from the coast into $[0, 1]$. Distances beyond $D_{\max} = 50\text{ km}$ saturate to $1.0$, preventing deep ocean areas from dominating feature gradients, while focusing high spatial sensitivity along the critical coastal margin where false alarms occur.
* **Terms & Parameters:**
  | Symbol | Name | Domain | Definition & Physical Unit |
  | :--- | :--- | :--- | :--- |
  | $\text{DistShore}_{[\text{norm}]}$ | Normalized Shore Feature | $[0.0, 1.0]$ | $0.0 = \text{Coastline}$, $1.0 = \text{Deep Ocean} \ge 50\text{ km}$. |
  | $D$ | Real Geodetic Distance | $\mathbb{R}^+$ | Metric distance from current pixel to closest shoreline ($[\text{m}]$). |
  | $D_{\max}$ | Normalization Horizon | Constant | Fixed at $50000.0\text{ m}$ ($50\text{ km}$). |

---

### 3.3 Ocean Bathymetry Normalization
$$\text{Bathy}_{[\text{norm}]} = \text{clip}\left(\frac{\text{depth}}{\text{depth}_{\max}}, 0.0, 1.0\right)$$

* **Conceptual Meaning:** Compresses seafloor depth into $[0, 1]$. Depths below $\text{depth}_{\max} = -2000\text{ m}$ saturate to $1.0$. This ensures the model is sensitive to shallow waters, coral reefs, and continental shelves ($0$ to $-200\text{ m}$) where ships fish and navigate.
* **Terms & Parameters:**
  | Symbol | Name | Domain | Definition & Physical Unit |
  | :--- | :--- | :--- | :--- |
  | $\text{Bathy}_{[\text{norm}]}$ | Normalized Depth Feature | $[0.0, 1.0]$ | $0.0 = \text{Sea Level (0 m)}$, $1.0 = \text{Abyssal Depth} \le -2000\text{ m}$. |
  | $\text{depth}$ | Ocean Floor Depth | $\mathbb{R}^-$ | Negative metric bathymetry ($[\text{m}]$). |
  | $\text{depth}_{\max}$ | Depth Clipping Threshold | Constant | Fixed at $-2000.0\text{ m}$. |

---

## 4. Deep Learning Architecture & Novelty (CGCAM)

```
[ Radar Features F_r (C x H x W) ] ---------> Key (K), Value (V) Projections
                                                        |
                                                        v
[ Context Features F_c (C_c x H x W) ] -----> Query (Q) Projection ---> [ Cross-Attention A ]
                                                                                |
                                                                                v
                                                                 F_att = A * V
                                                                                |
                                                                                v
[ F_out = F_r + gamma * Conv1x1(F_att) ] <--------------------------------------+
```

### 4.1 Linear Projections to Latent Space $d$
$$\mathbf{Q} = \mathbf{W}_Q \mathbf{F}_c, \quad \mathbf{K} = \mathbf{W}_K \mathbf{F}_r, \quad \mathbf{V} = \mathbf{W}_V \mathbf{F}_r$$

* **Conceptual Meaning:** Projects heterogenous radar features ($C$ channels) and geophysical context features ($C_c$ channels) into a shared intermediate latent subspace of dimension $d$ so their spatial correlation can be computed via dot-products.
* **Terms & Parameters:**
  | Symbol | Name | Shape / Domain | Definition & Conceptual Role |
  | :--- | :--- | :--- | :--- |
  | $\mathbf{F}_r$ | Radar Backbone Features | $\mathbb{R}^{C \times H \times W}$ | Latent representation extracted from $VV, VH$ radar channels. |
  | $\mathbf{F}_c$ | Context Encoder Features | $\mathbb{R}^{C_c \times H \times W}$ | Latent representation extracted from bathymetry & distance-to-shore. |
  | $\mathbf{W}_Q$ | Query Projection Weights | $\mathbb{R}^{d \times C_c}$ | $1 \times 1$ convolution weights projecting context features into Queries. |
  | $\mathbf{W}_K$ | Key Projection Weights | $\mathbb{R}^{d \times C}$ | $1 \times 1$ convolution weights projecting radar features into Keys. |
  | $\mathbf{W}_V$ | Value Projection Weights | $\mathbb{R}^{d \times C}$ | $1 \times 1$ convolution weights projecting radar features into Values. |
  | $\mathbf{Q}$ | Query Tensor | $\mathbb{R}^{d \times H \times W}$ | Query: *"Where does geography indicate navigable ocean far from cliffs?"*. |
  | $\mathbf{K}$ | Key Tensor | $\mathbb{R}^{d \times H \times W}$ | Key: *"Where did the radar observe high-energy reflections?"*. |
  | $\mathbf{V}$ | Value Tensor | $\mathbb{R}^{d \times H \times W}$ | Value: Raw radar feature content to be modulated and preserved. |
  | $d$ | Latent Projection Dimension | $\mathbb{N}^+$ | Intermediate feature channel capacity (e.g., $d = 64$). |
  | $H, W$ | Spatial Resolution | $\mathbb{N}^+$ | Height and width of the bottleneck feature map (e.g., $32 \times 32$). |

---

### 4.2 Scaled Dot-Product Spatial Cross-Attention
$$\mathbf{A}(i, j) = \text{Softmax}\left(\frac{\mathbf{Q}(i) \mathbf{K}(j)^T}{\sqrt{d}}\right)$$

* **Conceptual Meaning:** Computes a dense pairwise spatial compatibility matrix. If a radar pixel $j$ has bright intensity but context pixel $i$ indicates dry land or steep shoreline cliffs, the resulting attention weight is driven towards zero, effectively filtering out coastal false alarms.
* **Terms & Parameters:**
  | Symbol | Name | Domain | Definition & Conceptual Role |
  | :--- | :--- | :--- | :--- |
  | $\mathbf{A}(i, j)$ | Attention Weight | $[0, 1]$ | Normalized importance of radar position $j$ relative to context query $i$. |
  | $\mathbf{Q}(i)$ | Query Vector at Pixel $i$ | $\mathbb{R}^d$ | Geophysical context vector at flattened spatial index $i \in \{1, \dots, H \cdot W\}$. |
  | $\mathbf{K}(j)^T$ | Transposed Key Vector at $j$ | $\mathbb{R}^d$ | Radar feature vector at flattened spatial index $j \in \{1, \dots, H \cdot W\}$. |
  | $\sqrt{d}$ | Softmax Scaling Factor | $\mathbb{R}^+$ | Scaling factor preventing inner products from growing excessively large in high dimensions. |
  | $\text{Softmax}$ | Normalization Operator | Function | Ensures all weights along spatial dimension $j$ sum to $1.0$ ($\sum_j \mathbf{A}(i, j) = 1$). |

---

### 4.3 Gated Residual Modulation
$$\mathbf{F}_{\text{att}} = \mathbf{A} \mathbf{V}, \quad \mathbf{F}_{\text{out}} = \mathbf{F}_r + \gamma \cdot \text{Conv}_{1\times 1}(\mathbf{F}_{\text{att}})$$

* **Conceptual Meaning:** Multiplies attention weights by the radar value representation ($\mathbf{A} \mathbf{V}$) to produce context-filtered features $\mathbf{F}_{\text{att}}$, then injects them back into the original radar stream through a learnable residual gate $\gamma$.
* **Terms & Parameters:**
  | Symbol | Name | Shape / Domain | Definition & Conceptual Role |
  | :--- | :--- | :--- | :--- |
  | $\mathbf{F}_{\text{att}}$ | Attention-Weighted Features | $\mathbb{R}^{d \times H \times W}$ | Radar features filtered and modulated by geographic priors. |
  | $\mathbf{F}_{\text{out}}$ | Final Refined Output Map | $\mathbb{R}^{C \times H \times W}$ | The conditioned representation passed to the prediction heads. |
  | $\mathbf{F}_r$ | Original Radar Features | $\mathbb{R}^{C \times H \times W}$ | Identity shortcut connection preserving fine vessel details. |
  | $\gamma$ | Learnable Residual Gate | $\mathbb{R}$ | Scalar parameter initialized to $0.0$. Allows the network to gradually learn how much context to incorporate. |
  | $\text{Conv}_{1\times 1}$ | Channel Projection Layer | Function | Re-projects dimension $d$ back to the original backbone channel count $C$. |

---

## 5. Objective Function & Inference Optimization (Option 1: Unified 2-Channel CenterNet)

### 5.1 Multi-Channel Modified Focal Loss
$$\mathcal{L}_{\text{total}} = -\frac{1}{N_{\text{targets}}} \sum_{c \in \{0, 1\}} \sum_{x=1}^{W} \sum_{y=1}^{H} \begin{cases} 
(1 - \hat{Y}_{c, xy})^\alpha \log(\hat{Y}_{c, xy}) & \text{if } Y_{c, xy} = 1 \\
(1 - Y_{c, xy})^\beta (\hat{Y}_{c, xy})^\alpha \log(1 - \hat{Y}_{c, xy}) & \text{otherwise}
\end{cases}$$

* **Conceptual Meaning:** Solves extreme spatial class imbalance between millions of ocean background pixels and sparse vessel/structure centers across both semantic channels. It downweights easy ocean pixels by $(1 - \hat{Y}_{c, xy})^\alpha \approx 10^{-4}$ while dynamically preserving large gradients on misdetected targets.
* **Terms & Parameters:**
  | Symbol | Name | Domain | Definition & Conceptual Role |
  | :--- | :--- | :--- | :--- |
  | $\mathcal{L}_{\text{total}}$ | Total Objective Function | $\mathbb{R}^+$ | Multi-channel focal loss minimized via AdamW. |
  | $c$ | Semantic Channel Index | $\{0, 1\}$ | $c=0$: Mobile Vessels, $c=1$: Fixed Offshore Structures. |
  | $N_{\text{targets}}$ | Total Ground-Truth Targets | $\mathbb{N}^+$ | Total annotated maritime objects in the batch ($N = \sum_c N_c$). |
  | $x, y$ | Grid Pixel Coordinates | $\mathbb{N} \times \mathbb{N}$ | Spatial position on the $256 \times 256$ feature heatmap ($x \in [1, W], y \in [1, H]$). |
  | $Y_{c, xy}$ | Ground Truth Target Heatmap | $[0.0, 1.0]$ | Continuous Gaussian peak centered on ground-truth target centroid. |
  | $\hat{Y}_{c, xy}$ | Predicted Heatmap Activation | $[0.0, 1.0]$ | Model's sigmoid output probability at channel $c$ and position $(x, y)$. |
  | $\alpha$ | Modulating Exponent | Constant | Set to $2.0$. Suppresses loss on easy, well-classified background points. |
  | $\beta$ | Gaussian Distance Exponent | Constant | Set to $4.0$. Softens the loss for predictions slightly off-center ($Y_{c, xy} > 0$). |

---

### 5.2 Ground-Truth Gaussian Target Generation
$$Y_{c, xy} = \max_{k \in \mathcal{T}_c} \exp\left( -\frac{(x - x_k)^2 + (y - y_k)^2}{2\sigma^2} \right)$$

* **Conceptual Meaning:** Renders target peaks as continuous 2D Gaussians rather than discrete binary impulses ($1$-hot), avoiding harsh penalties on predictions within 1–2 pixels of the true center. Overlapping targets take the pointwise maximum.
* **Terms & Parameters:**
  | Symbol | Name | Domain | Definition & Conceptual Role |
  | :--- | :--- | :--- | :--- |
  | $\mathcal{T}_c$ | Target Set for Class $c$ | Set | Collection of annotated centroids $\{ (x_k, y_k) \}$ belonging to class $c$. |
  | $(x_k, y_k)$ | Annotated Ground-Truth Centroid | $\mathbb{N} \times \mathbb{N}$ | Exact coordinates of the $k$-th maritime target. |
  | $\sigma$ | Gaussian Standard Deviation | $\mathbb{R}^+$ | Spatial radius set to $\sigma = 2.0$ pixels (representing $\approx 20\text{ m}$ at $10\text{ m}$/pixel). |

---

### 5.3 Anchor-Free Peak Extraction via Local Max-Pooling (3x3 NMS)
$$\hat{\mathcal{P}}_c = \left\{ (x, y) \mid \hat{Y}_{c, xy} = \max_{(u, v) \in \mathcal{N}_3(x, y)} \hat{Y}_{c, uv} \quad \text{and} \quad \hat{Y}_{c, xy} \ge \tau \right\}$$

* **Conceptual Meaning:** Efficient post-processing replacing complex bounding-box NMS. A $3 \times 3$ max-pooling operation identifies local maxima; pixels that match the max-pooled map and exceed the confidence threshold $\tau$ are extracted as detected instances.
* **Terms & Parameters:**
  | Symbol | Name | Domain | Definition & Conceptual Role |
  | :--- | :--- | :--- | :--- |
  | $\hat{\mathcal{P}}_c$ | Set of Detected Detections | Set | Extracted coordinates of candidate objects for class $c$. |
  | $\mathcal{N}_3(x, y)$ | $3 \times 3$ Pixel Neighborhood | Set | The 8-connected local neighborhood around pixel $(x, y)$. |
  | $\tau$ | Confidence Threshold | $[0.0, 1.0]$ | Minimum activation required (e.g., $\tau = 0.3$). |

---

### 5.4 Metric Evaluation: Spatial Distance Matching & F1-Score
A predicted centroid $(\hat{x}, \hat{y}) \in \hat{\mathcal{P}}_c$ is matched as a **True Positive (TP)** to a ground-truth centroid $(x_k, y_k)$ if:
$$d_{\text{ground}}\left((\hat{x}, \hat{y}), (x_k, y_k)\right) = \sqrt{(\hat{x} - x_k)^2 + (\hat{y} - y_k)^2} \cdot s_{\text{pixel}} \le D_{\text{match}}$$
where $s_{\text{pixel}} = 10\text{ m/pixel}$ and $D_{\text{match}} = 100\text{ m}$ ($10\text{ pixels}$).

$$\text{Precision}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \quad F_{1, c} = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

* **Conceptual Meaning:** Standard maritime evaluation metric for Sentinel-1 SAR. A detection is declared valid if it falls within $100\text{ meters}$ of the ground truth, computing harmonic mean $F_1$ separately for vessels ($c=0$) and fixed structures ($c=1$).

---

## 6. Academic Grading Formulation

### 6.1 Final Course Grade Composition
$$\text{Final Grade} = 0.5 \cdot G_{\text{written}} + 0.5 \cdot G_{\text{project}}$$

* **Conceptual Meaning:** Standard arithmetic weighting governing the Computer Vision master's exam at Sapienza University of Rome. Both parts must meet the sufficiency threshold ($G \ge 18$) for the final grade to be recorded.
* **Terms & Parameters:**
  | Symbol | Name | Domain | Definition |
  | :--- | :--- | :--- | :--- |
  | $\text{Final Grade}$ | Registered Exam Grade | $[18, 30\text{L}]$ | Final grade recorded on Infostud. |
  | $G_{\text{written}}$ | Written Exam Grade | $[18, 32]$ | Score obtained on the 1-hour written exam (4 questions $\times$ 8 points). |
  | $G_{\text{project}}$ | Project Assessment | $[18, 30\text{L}]$ | Combined grade for code quality, novelty implementation, and oral defense. |
  | $0.5$ | Component Weight | Constant | Equal $50\% / 50\%$ distribution. |
