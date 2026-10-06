"""Checks for the scientific comparisons introduced by the CGCAM update."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from network import CGCAMModule, LocalContextGate, DarkVesselNet
from train import train_one_epoch
from optimizers import make_optimizer

torch.set_num_threads(2)


def test_zero_gamma_allows_gate_to_learn_then_context_gradients():
    torch.manual_seed(9)
    module = CGCAMModule(4, 3, 2, gamma_init=0.0)
    radar = torch.rand(2, 4, 4, 4)
    context = torch.rand(2, 3, 4, 4, requires_grad=True)
    opt = torch.optim.SGD(module.parameters(), lr=0.1)
    output = module(radar, context)
    torch.testing.assert_close(output, radar)
    output.square().mean().backward()
    assert module.gamma.grad.abs().item() > 0
    assert context.grad.abs().sum().item() == 0
    opt.step()
    opt.zero_grad()
    context.grad = None
    module(radar, context).square().mean().backward()
    assert context.grad.abs().sum().item() > 0


def test_local_gate_can_attenuate_and_amplify():
    module = LocalContextGate(2, 2, 2, gamma_init=0.1).eval()
    with torch.no_grad():
        module.gate_conv[-1].weight.zero_()
        module.gate_conv[-1].bias.copy_(torch.tensor([-2., 2.]))
    radar = torch.ones(1, 2, 5, 5)
    result = module(radar, torch.ones_like(radar))
    assert torch.all((result[:, 0] > 0.9) & (result[:, 0] < 1.0))
    assert torch.all((result[:, 1] > 1.0) & (result[:, 1] < 1.1))


def test_local_gate_retains_local_alignment_at_inference():
    torch.manual_seed(6)
    module = LocalContextGate(2, 2, 2, gamma_init=0.1).eval()
    radar = torch.ones(1, 2, 9, 9)
    a = torch.ones_like(radar)
    b = a.clone()
    b[:, :, 4, 4] += 2
    delta = (module(radar, a) - module(radar, b)).detach()
    assert delta[:, :, 3:6, 3:6].abs().sum() > 0
    delta[:, :, 3:6, 3:6] = 0
    assert delta.abs().sum() == 0


def test_shared_initialization_and_zero_gamma_match_sar():
    models = []
    for variant, mode, gamma in [("sar_only", "global", 0), ("cgcam", "global", 0),
                                 ("cgcam", "global", 0.1), ("cgcam", "local_gate", 0.1)]:
        torch.manual_seed(12)
        models.append(DarkVesselNet(variant, cgcam_mode=mode, gamma_init=gamma).eval())
    for key, value in models[0].state_dict().items():
        for model in models[1:]:
            torch.testing.assert_close(value, model.state_dict()[key])
    x = torch.rand(1, 5, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(models[0](x), models[1](x))


@pytest.mark.parametrize("mode", ["global", "local_gate"])
def test_checkpoint_reconstructs_mode_and_learned_gamma(tmp_path, mode):
    model = DarkVesselNet("cgcam_no_wind", cgcam_mode=mode, gamma_init=0.1).eval()
    with torch.no_grad():
        model.cgcam.gamma.fill_(-0.23)
    path = tmp_path/"checkpoint.pt"
    torch.save({"model_config": model.model_config(), "model_state_dict": model.state_dict()}, path)
    saved = torch.load(path, weights_only=True)
    restored = DarkVesselNet(**saved["model_config"]).eval()
    restored.load_state_dict(saved["model_state_dict"])
    x = torch.rand(1, 5, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(model(x), restored(x))
        # The no-wind ablation stays independent of the wind input in either mode.
        modified = x.clone()
        modified[:, 4] += 10
        torch.testing.assert_close(restored(x), restored(modified))
    assert restored.cgcam.gamma.item() == pytest.approx(-0.23)


def test_diagnostics_do_not_change_updates_and_record_first_step():
    torch.manual_seed(7)
    model = DarkVesselNet("cgcam", gamma_init=0.0)
    copy = DarkVesselNet("cgcam", gamma_init=0.0)
    copy.load_state_dict(model.state_dict())
    inputs = torch.rand(2, 5, 32, 32)
    target = torch.zeros(2, 2, 32, 32)
    target[:, :, 16, 16] = 1
    batches = [(inputs, target), (inputs, target)]
    stats = {}
    train_one_epoch(model, make_optimizer(model, "adamw", lr=1e-4), batches,
                    device="cpu", diagnostics=stats, diagnostics_every=1)
    train_one_epoch(copy, make_optimizer(copy, "adamw", lr=1e-4), batches, device="cpu")
    assert stats["samples"] == 2
    assert stats["first_sample"]["relative_branch_norm"] == 0
    assert stats["first_sample"]["context_grad_norm"] == 0
    assert stats["context_grad_norm_max"] > 0
    assert not model.cgcam._forward_hooks
    for a, b in zip(model.state_dict().values(), copy.state_dict().values()):
        torch.testing.assert_close(a, b)


def test_config_preserves_architecture_and_rejects_invalid_gamma(tmp_path, monkeypatch):
    import tuning
    monkeypatch.setattr(tuning, "dataset_fingerprint", lambda *args: {})
    args = tuning.parser().parse_args(["search", "--data-dir", str(tmp_path),
                                     "--cgcam-mode", "local_gate", "--gamma-init", "0.1"])
    cfg = tuning.build_cfg(args)
    assert cfg["cgcam_mode"] == "local_gate" and cfg["gamma_init"] == 0.1
    assert isinstance(tuning.build_model(dict(cfg, variant="cgcam", device="cpu")).cgcam, LocalContextGate)
    with pytest.raises(ValueError, match="finite"):
        DarkVesselNet("cgcam", gamma_init=float("nan"))
