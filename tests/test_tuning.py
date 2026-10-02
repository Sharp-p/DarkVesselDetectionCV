"""Focused checks for optimizer behavior, threshold math and evaluation isolation."""
import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optimizers import Lion, OPTIMIZERS, make_optimizer, parameter_groups
from network import DarkVesselNet
from tuning_metrics import DetectionCurve, at_frozen_thresholds, summarize
from tuning_runtime import collect_candidates, make_loader
from utils import match_detections_to_gt


def test_lion_two_updates_and_resume():
    p = torch.nn.Parameter(torch.tensor([1., -2.]))
    optimizer = Lion([p], lr=0.1, betas=(0.9, 0.99), weight_decay=0.2)
    p.grad = torch.tensor([2., -3.])
    optimizer.step()
    torch.testing.assert_close(p, torch.tensor([0.88, -1.86]))
    torch.testing.assert_close(optimizer.state[p]["exp_avg"], torch.tensor([0.02, -0.03]))
    q = torch.nn.Parameter(p.detach().clone())
    restored = Lion([q], lr=0.1)
    restored.load_state_dict(optimizer.state_dict())
    # Clone state tensors because load_state_dict can otherwise share them in memory.
    restored.state[q]["exp_avg"] = restored.state[q]["exp_avg"].clone()
    for param, opt in ((p, optimizer), (q, restored)):
        param.grad = torch.tensor([-0.1, 0.2])
        opt.step()
    torch.testing.assert_close(p, q)


def test_decay_excludes_gamma_bias_and_batchnorm():
    model = DarkVesselNet("cgcam")
    groups = parameter_groups(model, 0.1)
    decayed = {id(p) for p in groups[0]["params"]}
    exempt = {id(p) for p in groups[1]["params"]}
    assert not decayed & exempt
    assert decayed | exempt == {id(p) for p in model.parameters()}
    assert id(model.cgcam.gamma) in exempt
    for name, p in model.named_parameters():
        assert (id(p) in exempt) == (p.ndim <= 1 or name.endswith(".bias"))


@pytest.mark.parametrize("name", OPTIMIZERS)
def test_each_optimizer_takes_finite_step(name):
    model = torch.nn.Linear(3, 1)
    optimizer = make_optimizer(model, name, lr=1e-3, weight_decay=0 if name == "adam" else 0.01)
    before = model.weight.detach().clone()
    model(torch.ones(2, 3)).square().mean().backward()
    optimizer.step()
    assert torch.isfinite(model.weight).all()
    assert not torch.equal(before, model.weight)


def fixture_records():
    return [{"gt": [[(10, 10), (30, 30)], [(5, 5)]],
             "pred": [[(10, 10, 0.9, False), (50, 50, 0.8, True),
                       (30, 30, 0.2, False)], [(5, 5, 0.4, False)]]}]


def test_exact_f1_ap_and_class_thresholds():
    records = fixture_records()
    summary = summarize(records, distance_threshold=2)
    assert summary["best_by_channel"][0]["threshold"] == 0.2
    assert summary["best_by_channel"][0]["f1"] == pytest.approx(0.8)
    assert summary["best_by_channel"][1]["threshold"] == 0.4
    assert summary["ap_at_floor"][0] == pytest.approx(0.5 + 0.5 * 2/3)
    frozen = at_frozen_thresholds(records, [0.3, 0.3], distance_threshold=2)
    assert frozen[0]["tp"] == 1 and frozen[0]["fp"] == 1 and frozen[0]["fn"] == 1
    assert frozen[0]["near_shore_fp"] == 1


def test_cached_curve_matches_brute_force_greedy():
    rng = np.random.RandomState(4)
    records = []
    for _ in range(8):
        gt = [tuple(p) for p in rng.randint(0, 64, (5, 2))]
        preds = [(int(r), int(c), float(s), False) for (r,c), s in
                 zip(rng.randint(0, 64, (20, 2)), rng.rand(20))]
        records.append({"gt": [gt, []], "pred": [preds, []]})
    curve = DetectionCurve(records, 0, distance_threshold=10)
    for threshold in [0.01, 0.1, 0.3, 0.5, 0.9, 0.99]:
        counts = [0, 0, 0]
        for record in records:
            preds = [(r,c,s) for r,c,s,_ in record["pred"][0] if s >= threshold]
            tp, fp, fn, _ = match_detections_to_gt(preds, record["gt"][0], distance_threshold=10)
            counts = [x+y for x,y in zip(counts, (tp,fp,fn))]
        assert [curve.at(threshold)[k] for k in ("tp", "fp", "fn")] == counts


def test_score_ties_are_grouped_and_empty_class_not_fake_ap():
    records = [{"gt": [[(0,0)], []], "pred": [[(0,0,0.5,False), (20,20,0.5,True)], []]}]
    summary = summarize(records, distance_threshold=1)
    assert summary["ap_at_floor"] == [0.5, None]
    assert summary["best_by_channel"][0]["f1"] == pytest.approx(2/3)
    assert summary["best_by_channel"][1]["f1"] == 0


def test_logits_avoid_clamp_plateau_and_keep_zero_logit():
    class FixtureModel(torch.nn.Module):
        def forward(self, x, return_logits=False):
            logits = torch.full((1,2,9,9), -20.)
            logits[0,0,3:6,3:6] = torch.tensor([[10.,11.,10.],[11.,12.,11.],[10.,11.,10.]])
            logits[0,1,4,4] = 0.0
            heatmap = logits.sigmoid().clamp(1e-4, 1-1e-4)
            return heatmap, logits
    records, _ = collect_candidates(FixtureModel(), [(torch.zeros(1,5,9,9), torch.zeros(1,2,9,9))], "cpu", 0.3)
    assert len(records[0]["pred"][0]) == 1
    assert records[0]["pred"][1] == [(4,4,0.5,True)]


def test_network_optional_logits_does_not_change_default_api():
    torch.set_num_threads(2)
    model = DarkVesselNet("cgcam_no_wind").eval()
    x = torch.rand(1,5,32,32)
    original = x.clone()
    with torch.no_grad():
        p = model(x)
        q, logits = model(x, return_logits=True)
    torch.testing.assert_close(p, q)
    torch.testing.assert_close(q, logits.sigmoid().clamp(1e-4,1-1e-4))
    torch.testing.assert_close(x, original)


def test_background_sampling_seed_and_exact_patch_size():
    from data import DarkVesselDataset
    import pandas as pd
    dataset = DarkVesselDataset.__new__(DarkVesselDataset)
    dataset.df = pd.DataFrame([{"scene_id": "s", "detect_scene_row":128, "detect_scene_column":128}])
    dataset.scene_ids = ["s"]
    dataset.scene_shapes = {"s":(256,256)}
    dataset.sample_seed = 42
    assert dataset._build_samples() == dataset._build_samples()
    assert len(dataset._build_samples()) == 2


def test_calibration_rejects_test_cache(tmp_path):
    from tuning import run_calibrate
    path = tmp_path/"test.json.gz"
    with gzip.open(path, "wt") as f:
        json.dump({"split":"test", "candidate_floor":0.01, "records":fixture_records()}, f)
    with pytest.raises(ValueError, match="validation"):
        run_calibrate(SimpleNamespace(cache=path))


def test_search_config_does_not_fingerprint_test(tmp_path, monkeypatch):
    import tuning
    observed = []
    monkeypatch.setattr(tuning, "dataset_fingerprint", lambda cfg, splits: observed.extend(splits) or {})
    args = tuning.parser().parse_args(["search", "--data-dir", str(tmp_path)])
    tuning.build_cfg(args)
    assert observed == ["train", "val"]
