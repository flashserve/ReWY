# Installation

[Documentation](README.md) · [Usage](usage.md) · [Compatibility](compatibility.md)

## Choose an environment

| Goal | Recommended route |
| --- | --- |
| Use the backend in a CUDA training script | Fresh Linux environment + `scripts/install.sh cuda` |
| Use experimental Ascend dispatch | Matching CANN/TorchNPU/Triton-Ascend environment + `scripts/install.sh npu` |
| Reproduce the paper recipes | [Pinned Docker images](../containers/README.md) |
| Redraw reported figures | [CPU plotting](../benchmarks/README.md#render-the-figures), or Docker with `--figures-only` |

The library requires Python 3.11+. Source plotting requirements use Python 3.12+.
Device execution is supported on Linux. Windows can inspect source, run CPU
checks and invoke Docker for Linux containers.

## CUDA library

```bash
git clone https://github.com/flashserve/ReWY.git
cd ReWY
python3 -m venv .venv
source .venv/bin/activate
bash scripts/install.sh cuda
fdr verify --device cuda
```

The script installs Torch 2.9.1 from the CUDA 12.8 wheel index, Triton 3.5.1,
Transformers 5.21.0, einops 0.8.2 and FLA revision
`a59bc5df8c5841dce2169bb00e4027287197ba77`. It then installs this package and
checks the FLA source fingerprints. The measured device is A800-SXM4-80GB;
other GPUs need their own correctness and performance checks.

Keep this setup isolated from another project's environment: the installer
deliberately selects exact dependencies. If they are already installed,
`python -m pip install .` installs the library alone; it does not install or
upgrade Torch/FLA. Follow with `fdr doctor --check-sources`.

Clone-and-install is the documented installation method; it does not require
a PyPI release of `fast-delta-rule`.

## Ascend library

First activate an environment with matching CANN, TorchNPU and Triton-Ascend.
The installer checks device availability and uses that installed device stack;
it installs the pinned FLA and model dependencies.

```bash
bash scripts/install.sh npu
export FDR_EXPERIMENTAL_NPU=1
fdr verify --device npu
```

Current 9382 public operators require explicit initial state and final-state
output. The old native extension is not needed by public dispatch. Read
[9382 validation](ascend9382.zh-CN.md) before integrating a model. Ascend950 uses
the [dedicated preparation and experiment procedure](ascend950.zh-CN.md).

For exact paper reproduction, use the [9382 image](../containers/README.md).
It preserves the measured environment instead of resolving the library
installer's dependencies again, and requires CANN 9.0.0-beta.1 and the recorded
driver from the host. Paper images and library images serve different purposes.

## Verify and troubleshoot

```bash
fdr doctor                  # Package versions; no device experiment.
fdr doctor --check-sources  # Frozen FLA checks and device availability.
fdr verify --device cuda   # Device execution against a CPU FP64 reference.
```

| Symptom | Next step |
| --- | --- |
| `fdr: command not found` | Activate the environment; try `python -m fast_delta_rule --help`. |
| Missing/unsupported FLA sources | Use the platform installer in a fresh environment; version strings alone are insufficient. |
| First call is slow | Triton compiles on first use; warm up before timing. |
| Rejected shape, dtype or option | Check [the public envelope](compatibility.md); retain strict mode for measurements. |
| No optimized backward ran | Follow [dispatch troubleshooting](usage.md#confirm-the-backend-was-used). |
| Ascend image cannot initialize | Use the tested mounts, privileges and device mapping in [Docker instructions](../containers/README.md). |

For a library image built from the current checkout, see
[`containers/runtime/`](../containers/runtime/README.md).
