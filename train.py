"""
train.py

Implements the multi-task training loop, loss calculation (focal loss for detection,
cross-entropy for vessel classification), optimizer
and learning rate scheduling, and periodic validation logging.
"""

import os
import math
import torch
import torch.nn as nn
from globals import (
    BATCH_SIZE,
    PATCH_SIZE,
    INPUT_CHANNELS,
    NUM_CLASSES,
    GAUSSIAN_RADIAL_SIGMA,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    DEVICE,
    CHECKPOINT_DIR,
    set_seed,
)
from utils import focal_loss, save_checkpoint, load_checkpoint


""" Runs one training epoch over batches. Returns the mean loss across the epoch."""
def train_one_epoch(model, optimizer, batches, device=DEVICE, max_grad_norm=1.0):
    model.train()
    epoch_losses = []
    for inputs, targets in batches:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(inputs)
        loss = focal_loss(pred, targets)
        # Keep the pre-update safety check, using one scalar synchronization.
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise FloatingPointError("Non-finite training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()
        epoch_losses.append(loss_value)
    if not epoch_losses:
        raise ValueError("Training loader has no batches")
    return sum(epoch_losses) / len(epoch_losses)


""" Runs one validation pass over `batches`, no gradient updates. Returns the mean loss across the epoch. """
def validate(model, batches, device=DEVICE):
    model.eval()
    epoch_losses = []
    with torch.no_grad():
        for inputs, targets in batches:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            pred = model(inputs)
            loss = focal_loss(pred, targets)
            epoch_losses.append(loss)
    if not epoch_losses:
        raise ValueError("Validation loader is empty or has non-finite losses")
    epoch_losses = torch.stack(epoch_losses).cpu().tolist()
    if not all(math.isfinite(x) for x in epoch_losses):
        raise ValueError("Validation loader is empty or has non-finite losses")
    return sum(epoch_losses) / len(epoch_losses)


""" Saves a checkpoint only if val_loss improves on best_val_loss."""
def save_best_checkpoint(model, optimizer, epoch, val_loss, best_val_loss, checkpoint_dir=CHECKPOINT_DIR, filename="best_model.pt"):
    
    if val_loss < best_val_loss:
        path = save_checkpoint(model, optimizer, epoch=epoch, val_loss=val_loss, checkpoint_dir=checkpoint_dir, filename=filename)
        return val_loss, path
    return best_val_loss, None


"""Full train/val loop for num_epochs. Logs one clean line per epoch and saves a checkpoint only when val_loss improves, via save_best_checkpoint."""
def run_training(model, optimizer, train_batches, val_batches, num_epochs, scheduler=None, device=DEVICE, checkpoint_dir=CHECKPOINT_DIR, best_checkpoint_filename="best_model.pt"):
    
    best_val_loss = float("inf")
    best_checkpoint_path = None
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_batches, device=device)
        val_loss = validate(model, val_batches, device=device)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        improved_marker = ""
        best_val_loss, saved_path = save_best_checkpoint(
            model, optimizer, epoch, val_loss, best_val_loss,
            checkpoint_dir=checkpoint_dir, filename=best_checkpoint_filename,
        )
        if saved_path is not None:
            best_checkpoint_path = saved_path
            improved_marker = "  <- new best"

        print(f"epoch {epoch + 1}/{num_epochs} | "
              f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
              f"lr: {current_lr:.2e}{improved_marker}")
        if scheduler is not None:
            scheduler.step()

    return train_loss_history, val_loss_history, best_val_loss, best_checkpoint_path
