"""
utils.py

Provides utility functions, including evaluation metric computations (peak
extraction / NMS, centroid matching, precision, recall, F1-score, circle-IoU
localization proxy), SAR backscatter visualization routines, and checkpoint
saving/loading helpers.
"""

import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from globals import (
    FOCAL_ALPHA,
    FOCAL_BETA,
    PEAK_CONFIDENCE_THRESHOLD,
    NMS_KERNEL_SIZE,
    MATCH_DISTANCE_PIXELS,
    CHECKPOINT_DIR,
    DEVICE,
)


def focal_loss(pred, target, alpha=FOCAL_ALPHA, beta=FOCAL_BETA, eps=1e-4):
    pred = torch.clamp(pred, eps, 1.0 - eps)  # avoid log(0)
    pos_mask = target.eq(1).float()
    neg_mask = target.lt(1).float()
    pos_loss = -torch.log(pred) * torch.pow(1 - pred, alpha) * pos_mask
    neg_loss = (-torch.log(1 - pred) * torch.pow(pred, alpha)
                * torch.pow(1 - target, beta) * neg_mask)
    num_pos = pos_mask.sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    # normalize by number of ground-truth peaks; fall back to neg_loss alone
    # for background patches (no positives)
    # Keep the original all-background fallback without a CUDA-to-host scalar
    # branch on every batch. Clamp also keeps the unselected division finite.
    return torch.where(num_pos > 0, (pos_loss + neg_loss) / num_pos.clamp_min(1), neg_loss)


"""turns a predicted heatmap into a list of detections."""
def extract_peaks(heatmap, confidence_threshold=PEAK_CONFIDENCE_THRESHOLD,
                   kernel_size=NMS_KERNEL_SIZE):
    
    pad = kernel_size // 2
    pooled = F.max_pool2d(heatmap, kernel_size=kernel_size, stride=1, padding=pad)
    is_local_max = (pooled == heatmap).float()
    peaks = heatmap * is_local_max
    peaks = peaks * (peaks >= confidence_threshold).float()
    batch_size, num_channels = heatmap.shape[0], heatmap.shape[1]
    detections = []
    for b in range(batch_size):
        channel_dets = []
        for c in range(num_channels):
            rows, cols = torch.nonzero(peaks[b, c], as_tuple=True)
            confs = peaks[b, c, rows, cols]
            dets = [(int(r), int(cc), float(conf)) for r, cc, conf in zip(rows, cols, confs)]
            channel_dets.append(dets)
        detections.append(channel_dets)
    return detections


""" Greedy centroid matching, highest-confidence predictions matched first to their nearest unmatched ground-truth point, within distance_threshold."""
def match_detections_to_gt(pred_peaks, gt_peaks, distance_threshold=MATCH_DISTANCE_PIXELS):
    if len(gt_peaks) == 0:
        return 0, len(pred_peaks), 0, []
    if len(pred_peaks) == 0:
        return 0, 0, len(gt_peaks), []
    preds_sorted = sorted(pred_peaks, key=lambda d: -d[2])
    gt_matched = [False] * len(gt_peaks)
    tp, fp = 0, 0
    matched_pairs = []
    
    for (pr, pc, _conf) in preds_sorted:
        best_dist, best_idx = float("inf"), -1
        for i, (gr, gc) in enumerate(gt_peaks):
            if gt_matched[i]:
                continue
            dist = math.hypot(pr - gr, pc - gc)
            if dist < best_dist:
                best_dist, best_idx = dist, i

        if best_idx != -1 and best_dist <= distance_threshold:
            gt_matched[best_idx] = True
            tp += 1
            matched_pairs.append((pr, pc, gt_peaks[best_idx][0], gt_peaks[best_idx][1]))
        else:
            fp += 1
    fn = gt_matched.count(False)
    return tp, fp, fn, matched_pairs


def precision_recall_f1(tp, fp, fn, eps=1e-8):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1


"""IoU between two circles, used as a localization-quality proxy metric. This dataset has no ground-truth bounding boxes, only detection centroids, so classic box-IoU can't be computed directly. Representing each detection as a fixed-radius circle  gives a comparable overlap metric."""
def circle_iou(center_a, radius_a, center_b, radius_b):
    d = math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])
    r1, r2 = radius_a, radius_b

    if d >= r1 + r2:
        inter = 0.0  # circles don't overlap at all
    elif d <= abs(r1 - r2):
        inter = math.pi * min(r1, r2) ** 2  # smaller circle fully inside larger
    else:
        # standard circle-circle lens intersection formula
        alpha = 2 * math.acos(max(-1.0, min(1.0, (d**2 + r1**2 - r2**2) / (2 * d * r1))))
        beta = 2 * math.acos(max(-1.0, min(1.0, (d**2 + r2**2 - r1**2) / (2 * d * r2))))
        inter = (0.5 * r1**2 * (alpha - math.sin(alpha))
                 + 0.5 * r2**2 * (beta - math.sin(beta)))

    union = math.pi * r1**2 + math.pi * r2**2 - inter
    return inter / union if union > 0 else 0.0

""" Saves model + optimizer state, current epoch, and val_loss to 'epoch_{epoch}.pt' inside checkpoint_dir """
def save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_dir=CHECKPOINT_DIR, filename=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    if filename is None:
        filename = f"epoch_{epoch}.pt"
    path = os.path.join(checkpoint_dir, filename)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "model_config": model.model_config() if hasattr(model, "model_config") else None,
    }, path)
    return path


def load_checkpoint(model, optimizer, path, map_location=None):
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["val_loss"]


""" Loads only the model weights from a checkpoint and puts the model in eval mode on the target device."""
def load_model_for_inference(model, checkpoint_path, device=DEVICE):    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


""" Runs one forward pass in inference mode. Assumes model is already in eval mode """
def predict(model, inputs, device=DEVICE):
    inputs = inputs.to(device)
    with torch.no_grad():
        pred = model(inputs)
    return pred.detach().cpu()
