"""
train.py

Implements the multi-task training loop, loss calculation (focal loss for detection,
cross-entropy for vessel classification, smooth L1 for length estimation), optimizer
and learning rate scheduling, and periodic validation logging.
"""
