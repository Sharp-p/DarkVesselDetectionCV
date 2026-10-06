"""Sample fusion activity and gradients without retaining autograd graphs."""
import math

import torch


def grad_norm(parameters):
    grads = [p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return torch.stack(grads).sum().sqrt() if grads else None


class FusionDiagnostics:
    """Train-only hook; gradients are observed before global gradient clipping."""
    def __init__(self, model, every=10):
        self.model, self.every = model, every
        self.enabled = hasattr(model, "cgcam") and every > 0
        self.active = False
        self.rows = []
        self.current = None

    def __enter__(self):
        self.handle = self.model.cgcam.register_forward_hook(self._forward) if self.enabled else None
        return self

    def __exit__(self, *exc):
        if self.handle is not None:
            self.handle.remove()

    def begin_step(self, step):
        self.active = self.enabled and step % self.every == 0
        self.current = None
        self.step = step

    def _forward(self, module, inputs, output):
        if not self.active:
            return
        with torch.no_grad():
            radar = inputs[0].detach().float()
            delta = output.detach().float() - radar
            self.current = {"step": self.step,
                            "gamma_before": module.gamma.detach().reshape(()).clone(),
                            "relative_branch_norm": delta.norm() / radar.norm().clamp_min(1e-12)}

    def capture_gradients(self):
        if self.current is None:
            return
        gamma_grad = self.model.cgcam.gamma.grad
        zero = self.model.cgcam.gamma.detach().new_zeros(())
        self.current.update(
            gamma_grad=gamma_grad.detach().reshape(()).clone() if gamma_grad is not None else zero,
            context_grad_norm=grad_norm(self.model.context_stem.parameters()),
            fusion_grad_norm=grad_norm(p for name, p in self.model.cgcam.named_parameters() if name != "gamma"),
        )
        for name in ("context_grad_norm", "fusion_grad_norm"):
            if self.current[name] is None:
                self.current[name] = zero
        self.rows.append(self.current)

    def summary(self):
        if not self.rows:
            return {"enabled": False, "samples": 0}
        # One host transfer per epoch rather than five synchronization points
        # per sampled step. Only detached scalar tensors are retained.
        names = ("gamma_before", "relative_branch_norm", "gamma_grad",
                 "context_grad_norm", "fusion_grad_norm")
        values = torch.stack([torch.stack([r[k] for k in names]) for r in self.rows]).cpu().tolist()
        self.rows = [dict(zip(names, row), step=source["step"])
                     for source, row in zip(self.rows, values)]
        if not all(math.isfinite(v) for r in self.rows for v in r.values()):
            raise FloatingPointError("Non-finite fusion diagnostic or gradient")
        fields = ("relative_branch_norm", "context_grad_norm", "fusion_grad_norm")
        result = {"enabled": True, "samples": len(self.rows), "every_batches": self.every,
                  "branch_scope": "bottleneck_gate",
                  "gradient_stage": "before_clipping",
                  "gamma_start": self.rows[0]["gamma_before"],
                  "gamma_end": float(self.model.cgcam.gamma.detach()),
                  "gamma_grad_abs_mean": sum(abs(r["gamma_grad"]) for r in self.rows)/len(self.rows),
                  "first_sample": self.rows[0]}
        if hasattr(self.model, "context_gate_fine"):
            result["fine_gate_gamma_end"] = float(self.model.context_gate_fine.gamma.detach())
        for field in fields:
            result[field + "_mean"] = sum(r[field] for r in self.rows)/len(self.rows)
            result[field + "_max"] = max(r[field] for r in self.rows)
        return result
