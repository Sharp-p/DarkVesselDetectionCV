"""
network.py

Neural Network Architecture for Dark Vessel Detection via Multimodal SAR and CGCAM.
Option 1: Unified 2-Channel Anchor-Free CenterNet Heatmap (Vessel + Structure).

Key Components:
1. Radar backbone: Multi-scale residual CNN extracting radar representations (F_r).
2. Context encoder: Lightweight CNN extracting geophysical prior features (F_c).
3. CGCAMModule (Context-Guided Cross-Attention Module): Computes cross-attention between
   geophysical queries (Q) and radar keys/values (K, V) with a learnable gating residual.
4. Decoder: Upsamples the 32x32 bottleneck back to (256, 256).
5. Heatmap head: 2-channel 1x1 convolution predicting centroid heatmaps in [0, 1].

PROJECT_PLAN.md section 7 variants:
    Model A (SAR only):     DarkVesselNet(fusion_mode="sar_only")
    Model B (early fusion): DarkVesselNet(fusion_mode="early_fusion")
    Model C-wind:           DarkVesselNet(fusion_mode="cgcam_no_wind")
    Model C (proposed):     DarkVesselNet(fusion_mode="cgcam")

All four variants accept (B, 5, H, W) in the order
[VH_dB, VV_dB, bathymetry, distance_to_shore, wind_speed].

Inputs must already be normalized by data.py. Target generation, focal loss,
peak extraction, and detection metrics belong to the data/training/evaluation modules.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import globals as config
from globals import PATCH_SIZE, NUM_CLASSES, INPUT_CHANNELS, FUSION_MODE


# Contract is defined once in globals.py (5 channels: VH, VV, bathymetry,
# distance_to_shore, wind_speed). data.py must supply all five normalized channels.
FUSION_MODES = ("sar_only", "early_fusion", "cgcam_no_wind", "cgcam")


# -----------------------------------------------------------------------------
# 1. Basic Convolutional Blocks
# -----------------------------------------------------------------------------
class ConvBlock(nn.Module):
    """Standard Conv2d + BatchNorm2d + ReLU block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Residual Block with two 3x3 convolutions and an identity skip connection."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return self.relu(out)


# -----------------------------------------------------------------------------
# 2. Context-Guided Cross-Attention Module (CGCAM)
# -----------------------------------------------------------------------------
class CGCAMModule(nn.Module):
    """
    Context-Guided Cross-Attention Module (CGCAM).
    
    Decouples radar features from geophysical context:
    - Context features (bathymetry, distance-to-shore, wind speed) act as QUERIES (Q).
    - Radar features (VV, VH) act as KEYS (K) and VALUES (V).
    
    Implements FORMULARY.md sections 4.1-4.3 exactly. Suppression of coastal and
    wind-driven false alarms is a learned behavior to evaluate, not a fixed mask.
    """
    def __init__(self, radar_channels: int = 128, context_channels: int = 64, latent_dim: int = 64):
        super().__init__()
        if min(radar_channels, context_channels, latent_dim) <= 0:
            raise ValueError("CGCAM channel counts and latent_dim must be positive.")
        self.latent_dim = latent_dim
        self.scale = 1.0 / math.sqrt(latent_dim)

        # 1x1 linear projections into shared latent space d
        self.query_proj = nn.Conv2d(context_channels, latent_dim, kernel_size=1, bias=False)
        self.key_proj = nn.Conv2d(radar_channels, latent_dim, kernel_size=1, bias=False)
        self.value_proj = nn.Conv2d(radar_channels, latent_dim, kernel_size=1, bias=False)

        # Output projection back to radar feature channels
        self.out_proj = nn.Conv2d(latent_dim, radar_channels, kernel_size=1, bias=False)

        # Learnable residual gating scalar, initialized to 0.0 for stable warm-start
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, f_r: torch.Tensor, f_c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_r: Radar feature map (B, C_r, H, W)
            f_c: Context feature map (B, C_c, H, W)
        Returns:
            f_out: Context-modulated radar representation (B, C_r, H, W)
        """
        if f_r.ndim != 4 or f_c.ndim != 4:
            raise ValueError("CGCAM expects two (B, C, H, W) feature maps.")
        if f_r.shape[0] != f_c.shape[0] or f_r.shape[2:] != f_c.shape[2:]:
            raise ValueError("Radar and context features must have matching batch and spatial dimensions.")
        B, _, H, W = f_r.shape

        # 1. Project into latent space: (B, d, H, W) -> (B, N, d), N = H * W
        q = self.query_proj(f_c).flatten(2).transpose(1, 2)  # (B, N, d)
        k = self.key_proj(f_r).flatten(2)                  # (B, d, N)
        v = self.value_proj(f_r).flatten(2).transpose(1, 2)  # (B, N, d)

        # 2. Scaled Dot-Product Attention: (B, N, d) x (B, d, N) -> (B, N, N)
        attn_scores = torch.bmm(q, k) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, N, N)

        # 3. Aggregate values: (B, N, N) x (B, N, d) -> (B, N, d)
        f_att = torch.bmm(attn_weights, v)
        f_att = f_att.transpose(1, 2).reshape(B, self.latent_dim, H, W)

        # 4. Gated residual connection
        f_out = f_r + self.gamma * self.out_proj(f_att)
        return f_out


# -----------------------------------------------------------------------------
# 3. Complete Architecture: DarkVesselNet (Option 1)
# -----------------------------------------------------------------------------
class DarkVesselNet(nn.Module):
    """
    Unified 2-Channel Anchor-Free CenterNet with CGCAM.
    
    Input:
        (B, 5, 256, 256) where channels are:
        - 0: normalized VH_dB
        - 1: normalized VV_dB
        - 2: normalized bathymetry
        - 3: normalized distance_to_shore
        - 4: normalized wind_speed

        sar_only: only channels 0 and 1 reach the radar backbone.
        early_fusion: all five channels enter the backbone's first convolution.
        cgcam_no_wind: decoupled streams, with the context wind channel zeroed.
        cgcam: radar channels 0:2 and context channels 2:5 stay separate until CGCAM.
        Both CGCAM variants have identical parameter shapes; only wind routing
        differs, allowing a controlled comparison with the same initialization.
        Smaller/larger spatial dimensions divisible by 8 are supported for testing.
        
    Output:
        (B, 2, 256, 256) spatial probability heatmap:
        - Channel 0: Mobile Vessel Centroid Gaussian peaks in [0.0, 1.0]
        - Channel 1: Fixed Offshore Structure Centroid Gaussian peaks in [0.0, 1.0]
    """
    def __init__(
        self,
        fusion_mode: str = FUSION_MODE,
        num_classes: int = NUM_CLASSES,
    ):
        super().__init__()
        if fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {FUSION_MODES}; got {fusion_mode!r}.")
        if num_classes != 2:
            raise ValueError("The project contract requires two classes: vessel and structure.")
        self.fusion_mode = fusion_mode
        self.use_cgcam = fusion_mode in ("cgcam_no_wind", "cgcam")
        self.early_fusion = fusion_mode == "early_fusion"
        self.num_classes = num_classes

        # --- A. Radar Backbone (Channels 0 & 1: VH, VV) ---
        self.radar_stem = nn.Sequential(
            ConvBlock(in_channels=INPUT_CHANNELS if self.early_fusion else 2, out_channels=32,
                      kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),  # 256 -> 128
        )
        self.radar_stage1 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
        )
        self.radar_down = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)  # 128 -> 64
        self.radar_stage2 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
        )
        # A 32x32 bottleneck (the formulary's example) keeps N=1024: the dense
        # attention matrix has 16x fewer elements than attention at 64x64.
        # This spatial choice is shared by all four variants for ablation.
        self.radar_bottleneck = ConvBlock(128, 128, stride=2)  # 64 -> 32

        # --- B/C. Context Encoder and CGCAM (Proposed Model Only) ---
        # Baselines contain no unused context parameters or BatchNorm updates.
        if self.use_cgcam:
            self.context_stem = nn.Sequential(
                ConvBlock(3, 32, stride=2),   # Bathymetry, shore, wind; 256 -> 128
                ConvBlock(32, 64, stride=2),  # 128 -> 64
                ConvBlock(64, 64, stride=2),  # 64 -> 32
                ResidualBlock(64),
            )
            self.cgcam = CGCAMModule(radar_channels=128, context_channels=64, latent_dim=64)

        # --- D. Decoder & Upsampling Neck (32x32 -> 256x256) ---
        self.decoder_up0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 32 -> 64
            ConvBlock(128, 128),
        )
        self.decoder_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 64 -> 128
            ConvBlock(128, 64, kernel_size=3, stride=1, padding=1),
        )
        self.decoder_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # 128 -> 256
            ConvBlock(64, 32, kernel_size=3, stride=1, padding=1),
        )

        # --- E. Unified 2-Channel CenterNet Prediction Head ---
        self.heatmap_head = nn.Sequential(
            ConvBlock(32, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, self.num_classes, kernel_size=1, stride=1, padding=0),
        )

        # Bias initialization for focal loss stability: init bias to -2.19 (prob ~ 0.1)
        nn.init.constant_(self.heatmap_head[-1].bias, -2.19)

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        """
        Args:
            x: Normalized input (B, 5, H, W), shared by every fusion mode.
               H and W must be positive multiples of 8 (normally PATCH_SIZE).
        Returns:
            heatmap: (B, 2, H, W) float32 probabilities in [1e-4, 1.0 - 1e-4]
        """
        if x.ndim != 4 or x.shape[1] != INPUT_CHANNELS:
            raise ValueError(
                "Expected (B, 5, H, W) with channels "
                "[VH_dB, VV_dB, bathymetry, distance_to_shore, wind_speed]."
            )
        if any(size < 8 or size % 8 for size in x.shape[2:]):
            raise ValueError("Input height and width must be positive multiples of 8.")
        if not x.is_floating_point():
            raise TypeError("Expected normalized floating-point input.")
        x_radar = x if self.early_fusion else x[:, :2]

        # 1. Extract backbone features: (B, 128, 32, 32) for 256x256 patches.
        f_r = self.radar_stem(x_radar)
        f_r = self.radar_stage1(f_r)
        f_r = self.radar_down(f_r)
        f_r = self.radar_stage2(f_r)
        f_r = self.radar_bottleneck(f_r)

        # 2/3. Extract context and cross-attend only in the proposed model.
        if self.use_cgcam:
            x_context = x[:, 2:5]
            if self.fusion_mode == "cgcam_no_wind":
                # Preserve the three-channel encoder and the caller's input.
                # Zeroing only wind also removes its gradient from this variant.
                x_context = torch.cat((x[:, 2:4], torch.zeros_like(x[:, 4:5])), dim=1)
            f_c = self.context_stem(x_context)
            f_out = self.cgcam(f_r, f_c)
        else:
            f_out = f_r

        # 4. Decoder upsampling back to (256, 256)
        d0 = self.decoder_up0(f_out)  # (B, 128, 64, 64)
        d1 = self.decoder_up1(d0)    # (B, 64, 128, 128)
        d2 = self.decoder_up2(d1)     # (B, 32, 256, 256)

        # 5. Compute raw logits and apply sigmoid
        logits = self.heatmap_head(d2)  # (B, 2, 256, 256)
        # Keep the probability clamp representable under fp16 autocast, where
        # 1 - 1e-4 would otherwise round to 1 and permit log(0) in focal loss.
        heatmap = torch.sigmoid(logits.float())

        # Numerical clamping to prevent log(0) in Focal Loss
        heatmap = torch.clamp(heatmap, min=1e-4, max=1.0 - 1e-4)
        # Keep the existing training API; tuning detects peaks on unclamped logits.
        return (heatmap, logits.float()) if return_logits else heatmap


if __name__ == "__main__":
    print("Testing DarkVesselNet architecture contract...")
    dummy_input = torch.rand(4, INPUT_CHANNELS, PATCH_SIZE, PATCH_SIZE)
    for name, fusion_mode in (
        ("A / SAR only", "sar_only"),
        ("B / early fusion", "early_fusion"),
        ("C-wind / CGCAM without wind", "cgcam_no_wind"),
        ("C / CGCAM", "cgcam"),
    ):
        model = DarkVesselNet(fusion_mode=fusion_mode).eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert out.shape == (4, NUM_CLASSES, PATCH_SIZE, PATCH_SIZE), "Shape mismatch!"
        assert torch.isfinite(out).all(), "Non-finite heatmap!"
        assert ((out > 0) & (out < 1)).all(), "Invalid probability!"
        print(f"{name}: {tuple(dummy_input.shape)} -> {tuple(out.shape)}, "
              f"range=[{out.min().item():.4f}, {out.max().item():.4f}]")
    print("Architecture contract verified successfully!")
