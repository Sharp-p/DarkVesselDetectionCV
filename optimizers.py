"""Optimizer factory and explicit decay groups shared by training/tuning.

Lion implements Algorithm 2 in Chen et al., NeurIPS 2023:
https://arxiv.org/abs/2302.06675 (small independent implementation).
"""
import torch

OPTIMIZERS = ("adamw", "adam", "sgd", "rmsprop", "lion")

# v1 domains (pilot-v1/pilot-v2/full-v1 studies). Kept for reference/reproduction
# via a --search-space JSON; they are no longer the default.
SEARCH_SPACES_V1 = {
    "adamw": {"lr": [3e-5, 3e-4], "weight_decay": [0.0, 1e-4, 1e-3, 1e-2, 1e-1]},
    "adam": {"lr": [3e-5, 3e-4], "weight_decay": [0.0]},
    "sgd": {"lr": [1e-3, 1e-2], "weight_decay": [0.0, 1e-5, 1e-4, 1e-3]},
    "rmsprop": {"lr": [1e-4, 1e-3], "weight_decay": [0.0, 1e-5, 1e-4, 1e-3]},
    "lion": {"lr": [1e-5, 1e-4], "weight_decay": [0.0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]},
}

# v2 domains, refined from the 59 completed full-v1/pilot-v2 trials (30 epochs,
# batch 16, cosine, vessel-F1). See SEARCH_REFINEMENT.md for the evidence:
#  * AdamW/Adam: lr <= 4e-5 underfits in 30 epochs (0.55-0.66 F1); 5e-5..6e-5 gives
#    0.88-0.93; best runs are 1.5e-4..2.9e-4 and CGCAM's best sat on the old 3e-4
#    ceiling -> [5e-5, 1e-3].
#  * SGD: best 5.5e-3..9.9e-3, i.e. at the old 1e-2 ceiling -> [3e-3, 5e-2].
#  * RMSprop: flat 1.1e-4..7.6e-4, best 1.34e-4 near the floor -> [5e-5, 1e-3].
#  * Lion: all runs with lr <= 2.2e-5 peaked by epoch 1-6; the only reliable
#    runs were 9.5e-5..9.9e-5 at the old ceiling -> [5e-5, 1e-3].
#  * Weight decay showed no effect larger than replicate noise (+/-0.02 F1) for
#    any optimizer, so choices are trimmed to spend trials on the learning rate.
SEARCH_SPACES = {
    "adamw": {"lr": [5e-5, 1e-3], "weight_decay": [0.0, 1e-2, 1e-1, 3e-1]},
    "adam": {"lr": [5e-5, 1e-3], "weight_decay": [0.0]},
    "sgd": {"lr": [3e-3, 5e-2], "weight_decay": [0.0, 1e-4, 1e-3]},
    "rmsprop": {"lr": [5e-5, 1e-3], "weight_decay": [0.0, 1e-4, 1e-3]},
    "lion": {"lr": [5e-5, 1e-3], "weight_decay": [0.0, 1e-3, 1e-2, 1e-1]},
}

# Literature/reference starting points (used by v1).
DEFAULTS = {
    "adamw": {"lr": 1e-4, "weight_decay": 1e-2},
    "adam": {"lr": 1e-4, "weight_decay": 0.0},
    "sgd": {"lr": 3e-3, "weight_decay": 1e-4},
    "rmsprop": {"lr": 3e-4, "weight_decay": 0.0},
    "lion": {"lr": 3e-5, "weight_decay": 3e-2},
}

# Warm starts for v2: each optimizer's best full-v1 (sar_only) configuration,
# rounded. Enqueued once per optimizer so every optimizer is evaluated at its
# current incumbent (with the new multi-seed objective) before TPE explores.
WARM_STARTS = {
    "adamw": {"lr": 1.5e-4, "weight_decay": 1e-2},   # full-v1 #1: 0.947
    "adam": {"lr": 1e-4, "weight_decay": 0.0},       # full-v1 #0: 0.937
    "sgd": {"lr": 8e-3, "weight_decay": 0.0},        # full-v1 #9: 0.939
    "rmsprop": {"lr": 1.3e-4, "weight_decay": 1e-3},  # full-v1 #6: 0.942
    "lion": {"lr": 1e-4, "weight_decay": 1e-3},      # full-v1 #7: 0.920
}


class Lion(torch.optim.Optimizer):
    """Sign-momentum optimizer with decoupled weight decay; dense gradients only."""
    def __init__(self, params, lr=3e-5, betas=(0.9, 0.99), weight_decay=0.0):
        if lr <= 0 or weight_decay < 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError("Invalid Lion learning rate, betas, or weight decay")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")
                state = self.state[p]
                if not state:
                    state["exp_avg"] = torch.zeros_like(p)
                momentum = state["exp_avg"]
                p.mul_(1 - group["lr"] * group["weight_decay"])
                update = momentum.mul(beta1).add(grad, alpha=1 - beta1).sign_()
                p.add_(update, alpha=-group["lr"])
                momentum.mul_(beta2).add_(grad, alpha=1 - beta2)
        return loss


def parameter_groups(model, weight_decay):
    """Decay matrix/kernel weights, not bias/BatchNorm/scalar gate parameters."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay, "group_name": "weights"},
        {"params": no_decay, "weight_decay": 0.0, "group_name": "bias_norm_gate"},
    ]


def make_optimizer(model, name, lr, weight_decay=0.0):
    if name not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer {name!r}")
    if lr <= 0 or weight_decay < 0:
        raise ValueError("lr must be positive and weight_decay non-negative")
    if name == "adam" and weight_decay != 0:
        raise ValueError("The Adam control deliberately uses weight_decay=0")
    params = parameter_groups(model, weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, nesterov=True)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-8,
                                   momentum=0.0, centered=False)
    return Lion(params, lr=lr, betas=(0.9, 0.99))
