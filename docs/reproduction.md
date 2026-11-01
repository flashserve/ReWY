# Reproduction boundaries

[Documentation](README.md) · [Results](results.md)

## Choose what to reproduce

| Goal | Entry | Data and boundary |
| --- | --- | --- |
| Redraw the six current figures | `python benchmarks/plotting/make_figures.py --output-dir results/figures` | Public six-device aggregates; CPU only |
| Reproduce H100/H200 statefix1 | [Hopper image](../containers/hopper-statefix/README.zh-CN.md) | Low-level operator B and scoped model B |
| Rerun the paper's A800 / 9382 experiments | [Published images and launcher](../containers/README.md) | Recorded low-level B and separate full-model B recipes |
| Measure current package operators | `fdr benchmark` / `fdr reproduce` below | Public autograd synchronized eager B/FB |
| Run a single historical model or operator recipe | [Research launchers](../research/README.md) | Exact source version and boundary recorded with that study |
| Reproduce the separate Ascend950 study | [950 procedure](ascend950.zh-CN.md) | Prepared public B/FB and independent components |

The paper images collect new data and also replay the archived figures into a
separate directory. A replay is not a new hardware measurement. Historical
precision-limited controls retain their † labels. Published October 30 images retain their archived figure revision; use the checkout CPU command for the latest paper figures.

## Current public package

```bash
fdr verify --device cuda
fdr reproduce --suite release --device cuda --output results/run-001
```

The release suite first runs independent CPU FP64 correctness checks, then the
six paper shapes for four operators using matched recomputation, complete public
backward and 30 paired rounds (60 samples per implementation). It retains raw
samples, orders, correctness metrics, dispatch counts and an environment record.
The output directory must be new; previous results are never overwritten.
Geometric means are computed within each operator/device/boundary/save-policy.

This command is an explicit performance job, never run on package import or
installation. The default benchmark is synchronized eager wall-clock, with
forward outside the B interval and gradient conversions/allocation included.
`--boundary FB` includes public forward. Neither is a complete model training
step. JIT warmup is outside timing. The benchmark checks paired gradients before
timing and does not silently fall back to an original operator.

The release suite uses the paper's shape grid but is **not** a byte-identical
replay of historical low-level B/FB measurements. Those reports used different
boundaries, hardware-specific variants and sometimes experimental all-save
wrappers. In particular, obsolete GPU LOWP results cannot be reproduced as
precision-compliant claims by simply running an old kernel. This release exposes
only the current compliant GPU defaults. Archived reports must keep their own
version, hardware, precision, policy and timing labels.

## Current library container recipes

```bash
docker build -f containers/runtime/Dockerfile.cuda -t fast-delta-rule:0.1.0-cuda .
docker run --rm --gpus all fast-delta-rule:0.1.0-cuda verify --device cuda
docker run --rm --gpus all -v "$PWD/results:/workspace/results" \
  fast-delta-rule:0.1.0-cuda reproduce --suite release --output results/run-001
```

These recipes build the current **library** and use `fdr` as their entry point.
The separately [published paper images](../containers/README.md) have frozen
environments and use `run` / `replay` instead. Host device drivers are required
for both. For an Ascend library image, supply a matching toolchain base via
`--build-arg NPU_BASE=...`; see [runtime recipes](../containers/runtime/README.md).
