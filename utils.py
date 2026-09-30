"""
utils.py

Provides utility functions for the unified 2-channel CenterNet: 3x3 max-pooling peak
extraction (NMS), centroid matching within MATCH_DISTANCE_PIXELS, and per-channel
metrics (precision, recall, F1 for vessels and for structures). NO IoU or length
regression metrics: the model predicts centroids, not boxes or hull lengths.

Also provides SAR backscatter visualization routines and checkpoint save/load helpers.
"""
