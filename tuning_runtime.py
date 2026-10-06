"""Seeded loaders, synthetic smoke data, and sparse detection caching."""
import hashlib
import json
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


def make_loader(cfg, split, seed):
    if split not in ("train", "val", "test"):
        raise ValueError(split)
    if cfg["smoke"]:
        dataset = SyntheticDataset(split)
    else:
        from data import DarkVesselDataset
        dataset = DarkVesselDataset(cfg["scenes"][split], data_dir=cfg["data_dir"],
                                   labels_path=cfg["labels_path"], is_train=split == "train",
                                   sample_seed=cfg["data_seed"])
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
                      pin_memory=str(cfg["device"]).startswith("cuda"))


@torch.no_grad()
def collect_candidates(model, batches, device, min_score):
    """One forward per batch; cache sparse local maxima, GT and coastal flags.

    Local maxima are detected on raw logits (no clamp plateaus). Scores are
    sigmoid(logits), using a boolean mask so a zero logit is a valid detection.
    """
    model.eval()
    records, losses = [], []
    for inputs, targets in batches:
        heatmap, logits = model(inputs.to(device), return_logits=True)
        loss = focal_loss(heatmap, targets.to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation/evaluation loss")
        losses.append(float(loss))
        scores = logits.sigmoid()
        maxima = logits.eq(F.max_pool2d(logits, 3, stride=1, padding=1)) & scores.ge(min_score)
        maxima, scores, targets = maxima.cpu(), scores.cpu(), targets.cpu()
        for b in range(len(inputs)):
            record = {"pred": [], "gt": []}
            for c in range(2):
                coords = torch.nonzero(maxima[b, c], as_tuple=False)
                preds = [(int(r), int(k), float(scores[b, c, r, k]),
                          bool(inputs[b, 3, r, k] <= 0.1)) for r, k in coords]
                gt = [(int(r), int(k)) for r, k in torch.nonzero(targets[b, c].eq(1), as_tuple=False)]
                record["pred"].append(preds)
                record["gt"].append(gt)
            records.append(record)
    if not losses:
        raise ValueError("No evaluation batches")
    return records, sum(losses)/len(losses)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_digest():
    root = Path(__file__).resolve().parent
    names = ["network.py", "data.py", "globals.py", "train.py", "utils.py",
             "optimizers.py", "tuning.py", "tuning_metrics.py", "tuning_runtime.py",
             "fusion_diagnostics.py"]
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
