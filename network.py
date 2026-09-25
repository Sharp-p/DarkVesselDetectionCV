"""
network.py

Neural Network Architecture for Dark Vessel Detection via Multimodal SAR and CGCAM.
Option 1: Unified 2-Channel Anchor-Free CenterNet Heatmap (Vessel + Structure).

Key Components:
1. RadarBackbone: Multi-scale CNN extracting high-resolution radar representations (F_r).
2. ContextEncoder: Lightweight CNN extracting geophysical prior features (F_c).
3. CGCAM (Context-Guided Cross-Attention Module): Computes cross-attention between
   geophysical queries (Q) and radar keys/values (K, V) with a learnable gating residual.
4. DecoderNeck: Lightweight upsampling path restoring spatial dimensions to (256, 256).
5. UnifiedHeatmapHead: 2-channel 1x1 convolution predicting centroid heatmaps in [0, 1].
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from globals import PATCH_SIZE, INPUT_CHANNELS, NUM_CLASSES, USE_CGCAM


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
    - Context features (bathymetry, distance-to-shore) act as QUERIES (Q).
    - Radar features (VV, VH) act as KEYS (K) and VALUES (V).
    
    Computes spatial cross-attention to suppress high-backscatter coastal false alarms
    (cliffs, islands) where geophysical priors indicate non-navigable water or land.
    """
    def __init__(self, radar_channels: int = 128, context_channels: int = 64, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.scale = 1.0 / math.sqrt(latent_dim)

        # 1x1 linear projections into shared latent space d
        self.query_proj = nn.Conv2d(context_channels, latent_dim, kernel_size=1, bias=False)
        self.key_proj = nn.Conv2d(radar_channels, latent_dim, kernel_size=1, bias=False)
        self.value_proj = nn.Conv2d(radar_channels, latent_dim, kernel_size=1, bias=False)

        # Output projection back to radar feature channels
        self.out_proj = nn.Sequential(
            nn.Conv2d(latent_dim, radar_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(radar_channels),
        )

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
        B, C_r, H, W = f_r.shape
        N = H * W

        # 1. Project into latent space: (B, d, H, W) -> (B, N, d)
        q = self.query_proj(f_c).view(B, self.latent_dim, N).permute(0, 2, 1)  # (B, N, d)
        k = self.key_proj(f_r).view(B, self.latent_dim, N)                     # (B, d, N)
        v = self.value_proj(f_r).view(B, self.latent_dim, N).permute(0, 2, 1)  # (B, N, d)

        # 2. Scaled Dot-Product Attention: (B, N, d) x (B, d, N) -> (B, N, N)
        attn_scores = torch.bmm(q, k) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, N, N)

        # 3. Aggregate values: (B, N, N) x (B, N, d) -> (B, N, d)
        f_att = torch.bmm(attn_weights, v)
        f_att = f_att.permute(0, 2, 1).view(B, self.latent_dim, H, W)  # (B, d, H, W)

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
        (B, 4, 256, 256) where channels are:
        - 0: VH_dB
        - 1: VV_dB
        - 2: bathymetry
        - 3: distance_to_shore
        
    Output:
        (B, 2, 256, 256) spatial probability heatmap:
        - Channel 0: Mobile Vessel Centroid Gaussian peaks in [0.0, 1.0]
        - Channel 1: Fixed Offshore Structure Centroid Gaussian peaks in [0.0, 1.0]
    """
    def __init__(self, use_cgcam: bool = USE_CGCAM, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.use_cgcam = use_cgcam
        self.num_classes = num_classes

        # --- A. Radar Backbone (Channels 0 & 1: VH, VV) ---
        self.radar_stem = nn.Sequential(
            ConvBlock(in_channels=2, out_channels=32, kernel_size=3, stride=1, padding=1),
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

        # --- B. Geophysical Context Encoder (Channels 2 & 3: Bathymetry, Shore Distance) ---
        self.context_stem = nn.Sequential(
            ConvBlock(in_channels=2, out_channels=32, kernel_size=3, stride=2, padding=1),   # 256 -> 128
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            ResidualBlock(64),
        )

        # --- C. CGCAM Novelty Module ---
        if self.use_cgcam:
            self.cgcam = CGCAMModule(radar_channels=128, context_channels=64, latent_dim=64)

        # --- D. Decoder & Upsampling Neck (64x64 -> 256x256) ---
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
        self.heatmap_head[-1].bias.data.fill_(-2.19)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, 4, 256, 256)
        Returns:
            heatmap: (B, 2, 256, 256) with values clamped in [1e-4, 1.0 - 1e-4]
        """
        # Split input into radar (0, 1) and geophysical context (2, 3)
        x_radar = x[:, 0:2, :, :]    # (B, 2, 256, 256) -> [VH_dB, VV_dB]
        x_context = x[:, 2:4, :, :]  # (B, 2, 256, 256) -> [bathymetry, distance_to_shore]

        # 1. Extract radar features: (B, 128, 64, 64)
        f_r = self.radar_stem(x_radar)
        f_r = self.radar_stage1(f_r)
        f_r = self.radar_down(f_r)
        f_r = self.radar_stage2(f_r)

        # 2. Extract context features: (B, 64, 64, 64)
        f_c = self.context_stem(x_context)

        # 3. Apply CGCAM if enabled (Ablation toggle)
        if self.use_cgcam:
            f_out = self.cgcam(f_r, f_c)
        else:
            f_out = f_r

        # 4. Decoder upsampling back to (256, 256)
        d1 = self.decoder_up1(f_out)  # (B, 64, 128, 128)
        d2 = self.decoder_up2(d1)     # (B, 32, 256, 256)

        # 5. Compute raw logits and apply sigmoid
        logits = self.heatmap_head(d2)  # (B, 2, 256, 256)
        heatmap = torch.sigmoid(logits)

        # Numerical clamping to prevent log(0) in Focal Loss
        heatmap = torch.clamp(heatmap, min=1e-4, max=1.0 - 1e-4)
        return heatmap


if __name__ == "__main__":
    print("Testing DarkVesselNet architecture contract...")
    model = DarkVesselNet(use_cgcam=True, num_classes=NUM_CLASSES)
    model.eval()

    dummy_input = torch.randn(2, 4, 256, 256)
    with torch.no_grad():
        out = model(dummy_input)

    print(f"Model Input shape:  {dummy_input.shape}")
    print(f"Model Output shape: {out.shape} (Expected: (2, 2, 256, 256))")
    print(f"Output range: min={out.min().item():.4f}, max={out.max().item():.4f}")
    assert out.shape == (2, 2, 256, 256), "Shape mismatch with Option 1 contract!"
    print("Architecture contract verified successfully!")
