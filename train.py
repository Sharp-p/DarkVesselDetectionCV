"""
train.py

Training loop for the heatmap detector: one-epoch train/validate passes, best-checkpoint
saving, and run_training, which selects the checkpoint on either a validation score
(e.g. vessel F1) or the validation loss.
"""

import math
import torch
from globals import DEVICE, CHECKPOINT_DIR
from utils import focal_loss, save_checkpoint


def train_one_epoch(model, optimizer, batches, device=DEVICE, max_grad_norm=1.0):
    """One training epoch with gradient clipping. Returns the mean focal loss."""
    model.train()
    epoch_losses = []
    for inputs, targets in batches:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        pred = model(inputs)
        loss = focal_loss(pred, targets)
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


def validate(model, batches, device=DEVICE):
    """One validation pass, no gradient updates. Returns the mean focal loss."""
    model.eval()
    epoch_losses = []
    with torch.no_grad():
        for inputs, targets in batches:
            inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            pred = model(inputs)
            epoch_losses.append(focal_loss(pred, targets))
    if not epoch_losses:
        raise ValueError("Validation loader is empty")
    epoch_losses = torch.stack(epoch_losses).cpu().tolist()
    if not all(math.isfinite(x) for x in epoch_losses):
        raise ValueError("Validation loss is non-finite")
    return sum(epoch_losses) / len(epoch_losses)


def save_best_checkpoint(model, optimizer, epoch, score, best_score, checkpoint_dir=CHECKPOINT_DIR,
                         filename="best_model.pt"):
    """Saves a checkpoint only if `score` is lower than `best_score`.

    Returns (new_best_score, saved_path or None). run_training passes the validation loss,
    or the negative validation score, so lower is always better here. The "val_loss" field
    stored in the checkpoint holds that same value.
    """
    if score < best_score:
        path = save_checkpoint(model, optimizer, epoch=epoch, val_loss=score,
                               checkpoint_dir=checkpoint_dir, filename=filename)
        return score, path
    return best_score, None


def run_training(model, optimizer, train_batches, val_batches, num_epochs,
                 scheduler=None, device=DEVICE, checkpoint_dir=CHECKPOINT_DIR,
                 best_checkpoint_filename="best_model.pt",
                 max_grad_norm=1.0, val_score_fn=None):
    """Train for num_epochs and keep the best checkpoint.

    If val_score_fn(model, val_batches) -> float (higher is better) is given, the checkpoint
    is selected on it (e.g. validation F1). Otherwise it is selected on validation loss.
    Returns (train_hist, val_hist, best_value, best_checkpoint_path), where best_value is the
    best validation score (or the best validation loss if val_score_fn is None).
    """
    best_key = float("inf")          
    best_checkpoint_path = None
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, optimizer, train_batches, device=device,
                                     max_grad_norm=max_grad_norm)
        val_loss = validate(model, val_batches, device=device)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        val_score = val_score_fn(model, val_batches) if val_score_fn is not None else None
        key = val_loss if val_score is None else -val_score

        current_lr = optimizer.param_groups[0]["lr"]
        improved_marker = ""
        best_key, saved_path = save_best_checkpoint(
            model, optimizer, epoch, key, best_key,
            checkpoint_dir=checkpoint_dir, filename=best_checkpoint_filename,
        )
        if saved_path is not None:
            best_checkpoint_path = saved_path
            improved_marker = "  <- new best"

        score_text = f" | val_f1: {val_score:.3f}" if val_score is not None else ""
        print(f"epoch {epoch + 1}/{num_epochs} | "
              f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f}{score_text} | "
              f"lr: {current_lr:.2e}{improved_marker}")
        if scheduler is not None:
            scheduler.step()

    best_value = best_key if val_score_fn is None else -best_key
    return train_loss_history, val_loss_history, best_value, best_checkpoint_path
