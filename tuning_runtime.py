"""Seeded loaders, synthetic smoke data, sparse detection caching and fd hygiene."""
import errno
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import globals as config
from utils import focal_loss


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class SyntheticDataset(Dataset):
    """Tiny deterministic fixture, solely for installation/integration smoke tests."""
    def __init__(self, split, count=4):
        generator = torch.Generator().manual_seed({"train": 100, "val": 200, "test": 300}[split])
        self.x = torch.rand(count, 5, 32, 32, generator=generator) * 0.1
        self.y = torch.zeros(count, 2, 32, 32)
        yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
        for i in range(count):
            c, row, col = i % 2, 8 + i*3, 10 + i*2
            self.y[i, c] = torch.exp(-((yy-row)**2 + (xx-col)**2).float()/8)
            self.x[i, :2] += self.y[i, c] * 0.8

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


# ---------------------------------------------------------------------------
# File-descriptor hygiene. A long search runs dozens of trials in one process;
# every trial used to rebuild datasets and spawn persistent loader workers.
# ---------------------------------------------------------------------------
def fd_limits():
    """(soft, hard) RLIMIT_NOFILE, or (None, None) where unsupported."""
    try:
        import resource
        return resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ImportError, OSError, ValueError):
        return None, None


def raise_fd_limit(target=65536):
    """Raise the soft open-file limit towards the hard limit (never above it).

    Many Linux desktops default to a soft limit of 1024 with a far larger hard
    limit; raising the soft limit needs no privileges. Returns (old, new) soft.
    """
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = target if hard == resource.RLIM_INFINITY else min(target, hard)
        if soft != resource.RLIM_INFINITY and soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
            return soft, want
        return soft, soft
    except (ImportError, OSError, ValueError):
        return None, None


def open_fd_count():
    """Open descriptors of this process (Linux /proc, macOS /dev/fd) or None."""
    for path in ("/proc/self/fd", "/dev/fd"):
        try:
            return len(os.listdir(path))
        except OSError:
            continue
    return None


def is_fd_exhaustion(exc):
    """True for EMFILE/ENFILE anywhere in an exception chain, including the
    RuntimeError that DataLoader raises when it can no longer talk to workers."""
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, OSError) and exc.errno in (errno.EMFILE, errno.ENFILE):
            return True
        if "too many open files" in str(exc).lower():
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def shutdown_loader(loader):
    """Stop a DataLoader's persistent workers and pin-memory thread *now*.

    Relying on garbage collection is not enough: an exception traceback (pruning,
    NaN loss, OOM, Ctrl-C) keeps the training frame - and therefore the loader,
    its worker processes, pipes, queues and memmaps - alive into the next trial.
    """
    iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        try:
            shutdown()
        except Exception:  # best effort during cleanup
            pass
    loader._iterator = None


_DATASET_CACHE = {}


def _dataset(cfg, split):
    """Build each split's dataset once per process and reuse it across trials.

    Construction parses labels, reads context rasters and runs a distance
    transform; it is deterministic given (scenes, paths, data_seed), and the
    dataset holds no per-trial state (SAR memmaps are opened lazily inside the
    loader worker processes), so reuse does not change samples or augmentation.
    """
    if cfg["smoke"]:
        return SyntheticDataset(split)
    key = (split, cfg["data_dir"], cfg["labels_path"], tuple(cfg["scenes"][split]), cfg["data_seed"])
    if key not in _DATASET_CACHE:
        from data import DarkVesselDataset
        _DATASET_CACHE[key] = DarkVesselDataset(cfg["scenes"][split], data_dir=cfg["data_dir"],
                                                labels_path=cfg["labels_path"], is_train=split == "train",
                                                sample_seed=cfg["data_seed"])
    return _DATASET_CACHE[key]


def make_loader(cfg, split, seed):
    if split not in ("train", "val", "test"):
        raise ValueError(split)
    dataset = _dataset(cfg, split)
    limit = cfg.get(f"max_{split}_samples", 0)
    if limit:
        # The selected subset is stable across optimizers and training seeds.
        order = np.random.RandomState(cfg["data_seed"]).permutation(len(dataset))[:limit]
        dataset = Subset(dataset, order.tolist())
    if len(dataset) == 0:
        raise ValueError(f"No {split} samples")
    return DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=split == "train",
                      num_workers=cfg["workers"], worker_init_fn=seed_worker,
                      generator=torch.Generator().manual_seed(seed),
                      pin_memory=str(cfg["device"]).startswith("cuda"),
                      persistent_workers=cfg["workers"] > 0)


def iter_candidate_records(records):
    """Materialize legacy tuples only at the public/JSON serialization boundary."""
    for record in records:
        yield {
            "pred": [[(int(r), int(c), float(s), bool(near))
                      for r, c, s, near in (p.tolist() if isinstance(p, np.ndarray) else p)]
                     for p in record["pred"]],
            "gt": [[(int(r), int(c))
                    for r, c in (g.tolist() if isinstance(g, np.ndarray) else g)]
                   for g in record["gt"]],
        }


@torch.inference_mode()
def collect_candidates(model, batches, device, min_score, *, compact=False):
    """One forward per batch; cache sparse local maxima, GT and coastal flags.

    Local maxima are detected on raw logits (no clamp plateaus). Scores are
    sigmoid(logits), using a boolean mask so a zero logit is a valid detection.
    The tuning loop uses compact NumPy arrays; the default retains the original
    tuple-based API. Only sparse indices, scores and flags cross to the CPU.
    """
    model.eval()
    records, losses = [], []
    for inputs, targets in batches:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        heatmap, logits = model(inputs, return_logits=True)
        losses.append(focal_loss(heatmap, targets))
        scores = logits.sigmoid()
        maxima = logits.eq(F.max_pool2d(logits, 3, stride=1, padding=1)) & scores.ge(min_score)
        batch_size, channels, height, width = logits.shape
        area = height * width
        # One nonzero per whole batch, not per image/class. CUDA nonzero still
        # synchronizes, but no dense map or individual tensor scalar is copied.
        indices = torch.nonzero(maxima.reshape(-1), as_tuple=True)[0]
        shore_indices = (indices // (channels * area)) * area + indices % area
        sparse_scores = scores.reshape(-1)[indices]
        sparse_coastal = inputs[:, 3].reshape(-1)[shore_indices].le(0.1)
        gt_indices = torch.nonzero(targets.eq(1).reshape(-1), as_tuple=True)[0]
        # numpy does not support bfloat16; float32 preserves its scores exactly.
        score_dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
        flat = indices.cpu().numpy()
        values = sparse_scores.to(score_dtype).cpu().numpy()
        coastal = sparse_coastal.cpu().numpy()
        gt_flat = gt_indices.cpu().numpy()
        spatial, gt_spatial = flat % area, gt_flat % area
        predictions = np.column_stack((spatial // width, spatial % width, values, coastal))
        ground_truth = np.column_stack((gt_spatial // width, gt_spatial % width))
        # nonzero is lexicographically ordered, so slices preserve row/col ties
        # and retain empty images/classes without sorting or per-detection loops.
        boundaries = np.arange(batch_size * channels + 1) * area
        pred_offsets = np.searchsorted(flat, boundaries)
        gt_offsets = np.searchsorted(gt_flat, boundaries)
        for b in range(batch_size):
            record = {"pred": [], "gt": []}
            for c in range(channels):
                group = b * channels + c
                record["pred"].append(predictions[pred_offsets[group]:pred_offsets[group + 1]])
                record["gt"].append(ground_truth[gt_offsets[group]:gt_offsets[group + 1]])
            records.append(record)
    if not losses:
        raise ValueError("No evaluation batches")
    # Transfer scalar losses together, preserving the original mean-of-batches.
    losses = torch.stack(losses).cpu().tolist()
    if not all(math.isfinite(loss) for loss in losses):
        raise FloatingPointError("Non-finite validation/evaluation loss")
    return records if compact else list(iter_candidate_records(records)), sum(losses)/len(losses)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_digest():
    root = Path(__file__).resolve().parent
    names = ["network.py", "data.py", "globals.py", "train.py", "utils.py",
             "optimizers.py", "tuning.py", "tuning_metrics.py", "tuning_runtime.py"]
    return hashlib.sha256(json.dumps({n: file_sha256(root/n) for n in names}, sort_keys=True).encode()).hexdigest()


def dataset_fingerprint(cfg, splits):
    if cfg["smoke"]:
        return {"synthetic_fixture": 1}
    # Do not inspect test rasters during search; only the requested splits.
    files = {}
    for split in splits:
        for scene in cfg["scenes"][split]:
            for raster in ("VH_dB.tif", "VV_dB.tif", "bathymetry.tif", "owiMask.tif", "owiWindSpeed.tif"):
                path = Path(cfg["data_dir"])/scene/raster
                stat = path.stat()
                files[f"{scene}/{raster}"] = [stat.st_size, stat.st_mtime_ns]
    return {"labels_sha256": file_sha256(cfg["labels_path"]), "raster_size_mtime": files}
