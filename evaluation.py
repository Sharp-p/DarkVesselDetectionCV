"""
evaluation.py

Performs model inference and quantitative evaluation on the test set, generates visual
inspection plots (SAR ground-truth vs. predictions), and runs the comparative ablation
studies required for the Project-X submission.
"""
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from globals import (
    NUM_CLASSES,
    INPUT_CHANNELS,
    PEAK_CONFIDENCE_THRESHOLD,
    MATCH_DISTANCE_PIXELS,
    DEVICE,
    RESULTS_DIR,
    MAX_DISTANCE_SHORE_METERS,
    LEARNING_RATE,      
    WEIGHT_DECAY,       
    NUM_EPOCHS,         
)
from utils import (
    extract_peaks,
    match_detections_to_gt,
    precision_recall_f1,
    predict,
    load_model_for_inference,  
)
from train import run_training  

CHANNEL_NAMES = ["Vessel", "Fixed structure"]
SHORE_CHANNEL_INDEX = 3
NEAR_SHORE_KM_THRESHOLD = 5.0

"""Recovers ground-truth centroids from Gaussian-rendered target heatmaps: a GT centroid is any pixel where target == 1.0 exactly """
def gt_peaks_from_heatmap(target):
    
    batch_size, num_channels = target.shape[0], target.shape[1]
    gt = []
    for b in range(batch_size):
        channel_gt = []
        for c in range(num_channels):
            rows, cols = torch.nonzero(target[b, c].eq(1), as_tuple=True)
            channel_gt.append([(int(r), int(cc)) for r, cc in zip(rows, cols)])
        gt.append(channel_gt)
    return gt


"""Counts true positives, false positives and false negatives for one batch, separately per channel."""
def evaluate_batch(pred, target, confidence_threshold=PEAK_CONFIDENCE_THRESHOLD, distance_threshold=MATCH_DISTANCE_PIXELS):
    pred_dets = extract_peaks(pred, confidence_threshold=confidence_threshold)
    gt = gt_peaks_from_heatmap(target)

    num_channels = pred.shape[1]
    counts = [{"tp": 0, "fp": 0, "fn": 0} for _ in range(num_channels)]

    for b in range(pred.shape[0]):
        for c in range(num_channels):
            tp, fp, fn, _ = match_detections_to_gt(
                pred_dets[b][c], gt[b][c], distance_threshold=distance_threshold
            )
            counts[c]["tp"] += tp
            counts[c]["fp"] += fp
            counts[c]["fn"] += fn
    return counts


def merge_counts(list_of_batch_counts):
    num_channels = len(list_of_batch_counts[0])
    totals = [{"tp": 0, "fp": 0, "fn": 0} for _ in range(num_channels)]
    for batch_counts in list_of_batch_counts:
        for c in range(num_channels):
            for key in ("tp", "fp", "fn"):
                totals[c][key] += batch_counts[c][key]
    return totals


def counts_to_metrics(counts):
    metrics = []
    for channel_counts in counts:
        precision, recall, f1 = precision_recall_f1(
            channel_counts["tp"], channel_counts["fp"], channel_counts["fn"]
        )
        metrics.append({
            "tp": channel_counts["tp"],
            "fp": channel_counts["fp"],
            "fn": channel_counts["fn"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })
    return metrics


"""Runs the model over every (inputs, targets) batch, scores each batch,
    and returns final per-channel metrics computed from the summed counts."""
def evaluate_model(model, batches, device=DEVICE, confidence_threshold=PEAK_CONFIDENCE_THRESHOLD, distance_threshold=MATCH_DISTANCE_PIXELS):
    model.to(device)
    model.eval()

    all_counts = []
    for inputs, targets in batches:
        pred = predict(model, inputs, device=device)  # detached, on CPU
        all_counts.append(
            evaluate_batch(pred, targets.cpu(),
                           confidence_threshold=confidence_threshold,
                           distance_threshold=distance_threshold)
        )

    if not all_counts:
        raise ValueError("evaluate_model received no batches; nothing to evaluate")
    return counts_to_metrics(merge_counts(all_counts))


"""False positives within `shore_km_threshold` km of the coast """
def near_shore_false_positive_rate(pred, target, inputs, shore_channel_index=SHORE_CHANNEL_INDEX, shore_km_threshold=NEAR_SHORE_KM_THRESHOLD, max_shore_distance_km=MAX_DISTANCE_SHORE_METERS / 1000, confidence_threshold=PEAK_CONFIDENCE_THRESHOLD, distance_threshold=MATCH_DISTANCE_PIXELS):

    pred_dets = extract_peaks(pred, confidence_threshold=confidence_threshold)
    gt = gt_peaks_from_heatmap(target)
    num_channels = pred.shape[1]
    results = [{"total_fp": 0, "near_shore_fp": 0} for _ in range(num_channels)]
    for b in range(pred.shape[0]):
        shore_map = inputs[b, shore_channel_index]  # (H, W), normalized [0, 1]
        for c in range(num_channels):
            _tp, _fp, _fn, matched_pairs = match_detections_to_gt(
                pred_dets[b][c], gt[b][c], distance_threshold=distance_threshold
            )
            matched_pred_coords = {(pr, pc) for pr, pc, _gr, _gc in matched_pairs}
            fp_points = [(pr, pc) for (pr, pc, _conf) in pred_dets[b][c]
                         if (pr, pc) not in matched_pred_coords]

            results[c]["total_fp"] += len(fp_points)
            for (pr, pc) in fp_points:
                shore_km = shore_map[pr, pc].item() * max_shore_distance_km
                if shore_km <= shore_km_threshold:
                    results[c]["near_shore_fp"] += 1
    for r in results:
        r["near_shore_fp_rate"] = r["near_shore_fp"] / r["total_fp"] if r["total_fp"] > 0 else 0.0
    return results


"""Sums near-shore FP counts across batches, then recomputes the rate."""
def merge_near_shore_counts(list_of_batch_results):
    num_channels = len(list_of_batch_results[0])
    totals = [{"total_fp": 0, "near_shore_fp": 0} for _ in range(num_channels)]
    for batch_results in list_of_batch_results:
        for c in range(num_channels):
            totals[c]["total_fp"] += batch_results[c]["total_fp"]
            totals[c]["near_shore_fp"] += batch_results[c]["near_shore_fp"]
    for r in totals:
        r["near_shore_fp_rate"] = r["near_shore_fp"] / r["total_fp"] if r["total_fp"] > 0 else 0.0
    return totals


"""Runs the model over every (inputs, targets) batch and returns per-channel near-shore false-positive counts and rate, computed from summed totals."""
def evaluate_near_shore_false_positives(model, batches, device=DEVICE, shore_channel_index=SHORE_CHANNEL_INDEX, shore_km_threshold=NEAR_SHORE_KM_THRESHOLD, max_shore_distance_km=MAX_DISTANCE_SHORE_METERS / 1000, confidence_threshold=PEAK_CONFIDENCE_THRESHOLD, distance_threshold=MATCH_DISTANCE_PIXELS):
    
    model.to(device)
    model.eval()
    all_results = []
    for inputs, targets in batches:
        pred = predict(model, inputs, device=device)
        all_results.append(
            near_shore_false_positive_rate(
                pred, targets.cpu(), inputs.cpu(),
                shore_channel_index=shore_channel_index,
                shore_km_threshold=shore_km_threshold,
                max_shore_distance_km=max_shore_distance_km,
                confidence_threshold=confidence_threshold,
                distance_threshold=distance_threshold,
            )
        )
    if not all_results:
        raise ValueError("evaluate_near_shore_false_positives received no batches; nothing to evaluate")
    return merge_near_shore_counts(all_results)


"""Precision/recall/F1 per channel at each confidence threshold, for PR curves. Runs the model once per batch and re-scores the cached predictions at every threshold."""
def sweep_thresholds(model, batches, thresholds=None, device=DEVICE, distance_threshold=MATCH_DISTANCE_PIXELS):
    if thresholds is None:
        thresholds = [i / 20 for i in range(1, 20)]

    model.to(device)
    model.eval()
    cached = [(predict(model, inputs, device=device), targets.cpu())
              for inputs, targets in batches]
    if not cached:
        raise ValueError("sweep_thresholds received no batches; nothing to evaluate")

    results = []
    for t in thresholds:
        counts = merge_counts([
            evaluate_batch(pred, targets,
                           confidence_threshold=t,
                           distance_threshold=distance_threshold)
            for pred, targets in cached
        ])
        results.append({"threshold": t, "metrics": counts_to_metrics(counts)})
    return results


"""Pulls one channel's recalls, precisions, f1s out of sweep_thresholds output, in threshold order. """
def pr_points(sweep_results, channel):

    recalls = [r["metrics"][channel]["recall"] for r in sweep_results]
    precisions = [r["metrics"][channel]["precision"] for r in sweep_results]
    f1s = [r["metrics"][channel]["f1"] for r in sweep_results]
    return recalls, precisions, f1s


def _save_and_close(fig, save_path):
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)  # white background by default
    plt.close(fig)  
    return save_path

"""Precision-recall curve for one channel, one line per model."""
def plot_pr_curves(sweeps, channel, save_path, title=None):
    if not sweeps:
        raise ValueError("plot_pr_curves received no sweeps to plot")
    for label, sweep in sweeps.items():
        if not sweep:
            raise ValueError(f"sweep for '{label}' is empty")
        if not 0 <= channel < len(sweep[0]["metrics"]):
            raise ValueError(f"channel {channel} out of range for sweep '{label}'")
    channel_name = CHANNEL_NAMES[channel] if channel < len(CHANNEL_NAMES) else f"Channel {channel}"

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, sweep in sweeps.items():
        recalls, precisions, f1s = pr_points(sweep, channel)
        (line,) = ax.plot(recalls, precisions, marker="o", markersize=3, linewidth=2, label=label)
        best = max(range(len(f1s)), key=lambda i: f1s[i])
        ax.scatter([recalls[best]], [precisions[best]], marker="*", s=180,
                   color=line.get_color(), zorder=3)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_title(title or f"Precision-recall: {channel_name} (star = best F1)")
    ax.legend(loc="lower left")
    return _save_and_close(fig, save_path)


"""Train vs. validation loss per epoch."""
def plot_loss_curves(train_hist, val_hist, save_path, title="Training convergence"):
    if len(train_hist) == 0 or len(train_hist) != len(val_hist):
        raise ValueError("train_hist and val_hist must be non-empty and the same length")
    epochs = list(range(1, len(train_hist) + 1))
    best = min(range(len(val_hist)), key=lambda i: val_hist[i])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_hist, linewidth=2, label="Train loss")
    ax.plot(epochs, val_hist, linewidth=2, label="Validation loss")
    ax.axvline(best + 1, linestyle="--", color="gray", alpha=0.7,
               label=f"Best val (epoch {best + 1})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.legend()
    return _save_and_close(fig, save_path)


"""Splits one channel's predictions and ground truth into true positives, false positives, and false negatives, using the same greedy confidence-priority matching as match_detections_to_gt."""
def classify_detections(pred_dets, gt_points, distance_threshold=MATCH_DISTANCE_PIXELS):
    _tp, _fp, _fn, matched_pairs = match_detections_to_gt(
        pred_dets, gt_points, distance_threshold=distance_threshold)
    matched_pred_coords = {(pr, pc) for pr, pc, _gr, _gc in matched_pairs}
    matched_gt_coords = {(gr, gc) for _pr, _pc, gr, gc in matched_pairs}

    tp_points = [(pr, pc) for (pr, pc, _conf) in pred_dets if (pr, pc) in matched_pred_coords]
    fp_points = [(pr, pc) for (pr, pc, _conf) in pred_dets if (pr, pc) not in matched_pred_coords]
    fn_points = [pt for pt in gt_points if pt not in matched_gt_coords]
    return tp_points, fp_points, fn_points


"""Visual inspection plot for one sample: SAR backscatter with true positives, false positives, and false negatives overlaid."""
def plot_prediction_overlay(inputs, pred, target, save_path, channel=0, confidence_threshold=PEAK_CONFIDENCE_THRESHOLD, distance_threshold=MATCH_DISTANCE_PIXELS, sar_channel_index=0, title=None):
    pred_dets = extract_peaks(pred.unsqueeze(0), confidence_threshold=confidence_threshold)[0][channel]
    gt_points = gt_peaks_from_heatmap(target.unsqueeze(0))[0][channel]
    tp_points, fp_points, fn_points = classify_detections(pred_dets, gt_points, distance_threshold)

    sar_image = inputs[sar_channel_index].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(sar_image, cmap="gray")

    def _scatter(points, **kwargs):
        if points:
            ys, xs = zip(*points)
            ax.scatter(xs, ys, **kwargs)

    _scatter(tp_points, marker="o", s=80, facecolors="none", edgecolors="lime",
             linewidths=2, label=f"TP ({len(tp_points)})")
    _scatter(fp_points, marker="x", s=80, color="red",
             linewidths=2, label=f"FP ({len(fp_points)})")
    _scatter(fn_points, marker="o", s=80, facecolors="none", edgecolors="orange",
             linewidths=2, linestyle="--", label=f"FN ({len(fn_points)})")

    channel_name = CHANNEL_NAMES[channel] if channel < len(CHANNEL_NAMES) else f"Channel {channel}"
    ax.set_title(title or f"{channel_name} detections: TP / FP / FN")
    if tp_points or fp_points or fn_points:
        ax.legend(loc="upper right", framealpha=0.9)
    ax.axis("off")
    return _save_and_close(fig, save_path)


"""Saves one overlay PNG per sample in a batch, for scanning many patches quickly to pick good examples for the report/slides."""
def plot_prediction_overlays(inputs, pred, target, save_dir, channel=0, max_samples=None, confidence_threshold=PEAK_CONFIDENCE_THRESHOLD, distance_threshold=MATCH_DISTANCE_PIXELS, sar_channel_index=0, prefix="sample"):
    batch_size = inputs.shape[0]
    n = batch_size if max_samples is None else min(max_samples, batch_size)
    paths = []
    for b in range(n):
        save_path = os.path.join(save_dir, f"{prefix}_{b}.png")
        plot_prediction_overlay(
            inputs[b], pred[b], target[b], save_path, channel=channel,
            confidence_threshold=confidence_threshold, distance_threshold=distance_threshold,
            sar_channel_index=sar_channel_index,
        )
        paths.append(save_path)
    return paths


"""Trains and evaluates one model per fusion variant. For each variant: trains via run_training, reloads the best checkpoint, then runs evaluate_model, evaluate_near_shore_false_positives, and sweep_thresholds on the test set. Saves per-variant loss/PR curves plus one combined PR comparison figure into results_dir."""
def run_ablation(model_factory, train_batches, val_batches, test_batches, variants=("sar_only", "early_fusion", "cgcam"), num_epochs=NUM_EPOCHS, device=DEVICE, results_dir=RESULTS_DIR):

    def _resolve(batches, fusion_mode):
        return batches(fusion_mode) if callable(batches) else batches
    results = {}
    for fusion_mode in variants:
        model = model_factory(fusion_mode)
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        variant_train = _resolve(train_batches, fusion_mode)
        variant_val = _resolve(val_batches, fusion_mode)
        variant_test = _resolve(test_batches, fusion_mode)
        train_hist, val_hist, best_val_loss, checkpoint_path = run_training(
            model, optimizer, variant_train, variant_val, num_epochs,
            scheduler=scheduler, device=device,
            checkpoint_dir=os.path.join(results_dir, "checkpoints"),
            best_checkpoint_filename=f"best_{fusion_mode}.pt",
        )
        if checkpoint_path is not None:
            load_model_for_inference(model, checkpoint_path, device=device)
        test_metrics = evaluate_model(model, variant_test, device=device)
        near_shore_metrics = evaluate_near_shore_false_positives(model, variant_test, device=device)
        sweep = sweep_thresholds(model, variant_test, device=device)

        plot_loss_curves(train_hist, val_hist,
                          os.path.join(results_dir, f"loss_curves_{fusion_mode}.png"),
                          title=f"Training convergence: {fusion_mode}")
        plot_pr_curves({fusion_mode: sweep}, channel=0,
                        save_path=os.path.join(results_dir, f"pr_vessel_{fusion_mode}.png"))

        results[fusion_mode] = {
            "train_loss_history": train_hist,
            "val_loss_history": val_hist,
            "best_val_loss": best_val_loss,
            "checkpoint_path": checkpoint_path,
            "test_metrics": test_metrics,
            "near_shore_metrics": near_shore_metrics,
            "sweep": sweep,
        }

    plot_pr_curves({fm: r["sweep"] for fm, r in results.items()}, channel=0,
                    save_path=os.path.join(results_dir, "pr_vessel_comparison.png"),
                    title="Precision-recall comparison: Vessel detection")
    table = ablation_table(results)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "ablation_table.json"), "w") as f:
        json.dump(table, f, indent=2)

    return results, table


def ablation_table(results, channel=0):
    rows = []
    for fusion_mode, r in results.items():
        m = r["test_metrics"][channel]
        ns = r["near_shore_metrics"][channel]
        rows.append({
            "fusion_mode": fusion_mode,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            "near_shore_fp": ns["near_shore_fp"],
            "near_shore_fp_rate": ns["near_shore_fp_rate"],
        })
    return rows 
