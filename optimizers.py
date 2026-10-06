"""Optimizer factory and explicit decay groups shared by training/tuning.

Lion implements Algorithm 2 in Chen et al., NeurIPS 2023:
https://arxiv.org/abs/2302.06675 (small independent implementation).
"""
import torch

OPTIMIZERS = ("adamw", "adam", "sgd", "rmsprop", "lion")
# Initial search domains, not claims about optimal settings for this dataset.
SEARCH_SPACES = {
    "adamw": {"lr": [3e-5, 3e-4], "weight_decay": [0.0, 1e-4, 1e-3, 1e-2, 1e-1]},
    "adam": {"lr": [3e-5, 3e-4], "weight_decay": [0.0]},
    "sgd": {"lr": [1e-3, 1e-2], "weight_decay": [0.0, 1e-5, 1e-4, 1e-3]},
    "rmsprop": {"lr": [1e-4, 1e-3], "weight_decay": [0.0, 1e-5, 1e-4, 1e-3]},
    "lion": {"lr": [1e-5, 1e-4], "weight_decay": [0.0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1]},
}
DEFAULTS = {
    "adamw": {"lr": 1e-4, "weight_decay": 1e-2},
    "adam": {"lr": 1e-4, "weight_decay": 0.0},
    "sgd": {"lr": 3e-3, "weight_decay": 1e-4},
    "rmsprop": {"lr": 3e-4, "weight_decay": 0.0},
    "lion": {"lr": 3e-5, "weight_decay": 3e-2},
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
