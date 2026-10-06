"""Behavior checks for vectorized extraction, matching, and loader optimizations."""
from pathlib import Path
import sys
import math

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tuning_metrics import DetectionCurve
from tuning_runtime import collect_candidates, make_loader
from data import draw_gaussian_peak
from utils import focal_loss


@pytest.mark.parametrize("positive", [False, True])
def test_branchless_focal_preserves_loss_and_gradient(positive):
    torch.manual_seed(12)
    a = torch.rand(2, 2, 9, 9, requires_grad=True)
    b = a.detach().clone().requires_grad_()
    target = torch.zeros_like(a)
    if positive:
        target[:, :, 4, 4] = 1
        target[:, :, 4, 5] = 0.8
    pred = b.clamp(1e-4, 1-1e-4)
    pos = target.eq(1).float()
    neg = target.lt(1).float()
    lp = (-pred.log()*(1-pred).pow(2)*pos).sum()
    ln = (-(1-pred).log()*pred.pow(2)*(1-target).pow(4)*neg).sum()
    reference = (lp+ln)/pos.sum() if positive else ln
    actual = focal_loss(a, target)
    actual.backward()
    reference.backward()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)


def reference_counts(records, channel, threshold, distance):
    tp = fp = coastal = total = 0
    for record in records:
        gt = record["gt"][channel]
        total += len(gt)
        unused = set(range(len(gt)))
        for row, col, score, near in sorted(record["pred"][channel], key=lambda p: (-p[2], p[0], p[1])):
            if score < threshold:
                continue
            i = min(unused, key=lambda i: (math.hypot(row-gt[i][0], col-gt[i][1]), i), default=None)
            if i is not None and math.hypot(row-gt[i][0], col-gt[i][1]) <= distance:
                unused.remove(i)
                tp += 1
            else:
                fp += 1
                coastal += bool(near)
    return tp, fp, total-tp, coastal


def test_indexed_matching_equals_reference_with_ties_and_empty_patches():
    rng = np.random.RandomState(41)
    records = []
    for _ in range(30):
        gt, pred = [], []
        for channel in range(2):
            gt.append([tuple(x) for x in rng.randint(0, 64, (rng.randint(0, 30), 2)).tolist()])
            pred.append([(int(r), int(c), float(s), bool(n)) for (r, c), s, n in zip(
                rng.randint(0, 64, (150, 2)), np.round(rng.rand(150), 2), rng.randint(0, 2, 150))])
        records.append({"gt": gt, "pred": pred})
    for channel in range(2):
        for radius in (0, 5, 20):
            curve = DetectionCurve(records, channel, radius)
            for threshold in (0, 0.1, 0.3, 0.55, 0.9, 1, 1.01):
                row = curve.at(threshold)
                assert tuple(row[k] for k in ("tp", "fp", "fn", "near_shore_fp")) == reference_counts(records, channel, threshold, radius)


def test_distance_boundary_and_nearest_unmatched_fallback():
    records = [{"gt": [[(0, 0), (0, 6)], []],
                "pred": [[(0, 0, 0.9, False), (3, 2, 0.8, True), (3, 10, 0.7, True)], []]}]
    curve = DetectionCurve(records, 0, 5)
    assert curve.at(0.1)["tp"] == 2
    assert curve.at(0.1)["fp"] == 1
    assert curve.best_f1(0.1, 0.95)["threshold"] == 0.8


def test_vectorized_extraction_matches_scalar_reference():
    torch.manual_seed(3)
    logits = torch.randn(2, 2, 20, 20)
    x = torch.rand(2, 5, 20, 20)
    # Exact shore threshold (float32 comparison must stay identical).
    x[:, 3, 3:6, 3:6] = 0.1
    y = torch.zeros(2, 2, 20, 20)
    y[:, :, 5, 5] = 1
    class Model(torch.nn.Module):
        def forward(self, x, return_logits=False):
            return logits.sigmoid(), logits
    records, _ = collect_candidates(Model(), [(x, y)], "cpu", 0.1)
    scores = logits.sigmoid()
    maxima = logits.eq(torch.nn.functional.max_pool2d(logits, 3, stride=1, padding=1)) & scores.ge(0.1)
    for b in range(2):
        for c in range(2):
            expected = [(int(r), int(k), float(scores[b, c, r, k]), bool(x[b, 3, r, k] <= 0.1))
                        for r, k in torch.nonzero(maxima[b, c])]
            assert records[b]["pred"][c] == expected
            assert records[b]["gt"][c] == [(5, 5)]


@pytest.mark.parametrize("sigma", [1.5, 2., 3.])
def test_cached_gaussian_matches_original_at_edges_and_overlap(sigma):
    result, expected = np.zeros((25, 25), np.float32), np.zeros((25, 25), np.float32)
    for cx, cy in [(0, 0), (24, 24), (12, 12), (13, 12), (-2, 3)]:
        draw_gaussian_peak(result, cx, cy, sigma)
        radius = int(3*sigma)
        x0, x1 = max(0, cx-radius), min(25, cx+radius+1)
        y0, y1 = max(0, cy-radius), min(25, cy+radius+1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        gaussian = np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sigma*sigma)).astype(np.float32)
        expected[y0:y1, x0:x1] = np.maximum(expected[y0:y1, x0:x1], gaussian)
    np.testing.assert_array_equal(result, expected)


def test_worker_zero_ignores_prefetch_options():
    cfg = dict(smoke=True, data_seed=42, batch_size=2, workers=0, device="cpu",
               persistent_workers=True, prefetch_factor=2)
    loader = make_loader(cfg, "train", 42)
    assert not loader.persistent_workers
    assert len(list(loader)) == 2


def test_persistent_loader_has_repeatable_order_across_independent_runs():
    # PyTorch's process tensor transport needs local Unix-domain sockets.
    # Some hosted executors prohibit them; skip explicitly, never hang silently.
    import socket
    try:
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    except PermissionError:
        pytest.skip("Executor prohibits Unix sockets required by multiprocessing tensor transport")
    else:
        probe.close()
    cfg = dict(smoke=True, data_seed=42, batch_size=2, workers=1, device="cpu",
               persistent_workers=True, prefetch_factor=2, loader_timeout=20)
    def run():
        loader = make_loader(cfg, "train", 42)
        result = [torch.cat([x for x, _ in loader]) for _ in range(2)]
        del loader
        return result
    for a, b in zip(run(), run()):
        torch.testing.assert_close(a, b)
