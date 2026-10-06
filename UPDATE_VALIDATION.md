# Validation of the matched-recall update

- Full test suite: **45 passed, 1 skipped**. The skip is the existing hosted-environment restriction on Unix sockets for loader workers.
- Two one-epoch synthetic wind/no-wind runs completed with the same RMSprop settings and the recall-constrained objective. Automatic validation reports were generated.
- Both selected checkpoints were evaluated with frozen validation recall cutoffs; the paired test report was generated successfully.
- Architecture is unchanged from the CPU-optimized archive; the finer-resolution experiment was removed.
- No real SAR training or GPU benchmark was performed; accuracy improvements are unverified.

# Validation of the CGCAM update

The results below describe the initial architecture update. The subsequent CPU
performance update is documented in `PERFORMANCE_README.md` and
`cpu_benchmark.json`, with additional checks in `tests/test_performance.py`.

- `python -m pytest -q tests`: **23 passed**.
- Controlled comparison launcher: all eight configurations completed a one-epoch,
  one-seed synthetic CPU smoke run, including original attention, gamma 0.1, and
  local gating with and without wind.
- Frozen-threshold evaluation successfully reloaded a local-gate checkpoint and
  preserved validation-selected thresholds.
- Fixed-configuration repeat successfully trained a new seed using the saved
  local-gate settings.
- All Python source files parsed successfully; archive integrity was checked.

Focused tests cover zero-gamma gradient flow, attenuation and amplification,
locality at inference, shared initialization, checkpoint reconstruction and
learned gamma, wind ablation, diagnostics without changing training updates,
optimizer behavior, exact F1 threshold searches, and validation/test isolation.

Environment: Python 3.12, PyTorch 2.14.1+cpu, Optuna 4.5.0. No real SAR dataset
training or GPU performance measurements were run for this update. Synthetic
scores do not show whether the new architecture improves detection performance.
The existing server implementation was retained; the new smoke comparison used
direct SQLite storage rather than launching the gRPC server/dashboard.
