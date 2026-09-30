"""
train.py

Implements the training loop for the unified 2-channel CenterNet heatmap head.
Single objective: multi-channel modified focal loss (FOCAL_ALPHA/FOCAL_BETA from
globals.py) over the two semantic channels (0 = mobile vessel, 1 = fixed offshore
structure). NO auxiliary classification or length-regression heads exist: detection
and discrimination live in the heatmap channels, so there is no loss balancing.

Also provides optimizer setup (AdamW, weight decay), learning-rate scheduling,
gradient clipping, checkpointing, and periodic validation logging. Trains any of
the four ablation variants by passing FUSION_MODE to DarkVesselNet.
"""
