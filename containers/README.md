# Paper reproduction images

[Documentation](../docs/README.md) · [Current library images](runtime/README.md)

The October 30 images contain that paper revision's retained source, frozen FLA, measured
Python environments, raw paper evidence and figure generators. No model-weight
download or package installation is needed at run time. Models are initialized
with the recorded seeds, as in the experiments.

The November 1 paper adds [H100/H200 statefix1 results and image](hopper-statefix/README.zh-CN.md).
These use a separate pinned image from the October 30 A800/9382 images. To redraw the
latest six-device figures from this checkout on CPU:

```bash
python -m pip install -r benchmarks/requirements.txt
python benchmarks/plotting/make_figures.py --output-dir results/paper-figures
```

| Platform | Docker Hub | Tag | Architecture | Experiment environment |
| --- | --- | --- | --- | --- |
| A800-SXM4-80GB | [residual-wy-a800](https://hub.docker.com/r/zhangyi2026/residual-wy-a800) | `20261030-paper` | linux/amd64 | Python 3.12.3, Torch 2.9.1+cu128, Triton 3.5.1 |
| Ascend910_9382, 64 GiB | [residual-wy-ascend9382](https://hub.docker.com/r/zhangyi2026/residual-wy-ascend9382) | `20261030-paper` | linux/arm64 | Python 3.11.17, Torch 2.7.1+cpu, TorchNPU 2.7.1.post4, Triton-Ascend 3.2.1 |

The A800 image uses the original `/workspace/env` virtual environment, including
its CUDA 12.8 PyTorch libraries. The larger upstream PyTorch base supplies Ubuntu
24.04 and Python 3.12; its system PyTorch is not selected by the entry point.
The Ascend image preserves the measured virtual environment and Python paths.
It includes the historical KDA native library and its source; the driver and
CANN toolkit must come from the host. The recorded toolkit is **CANN
9.0.0-beta.1**, driver/npu-smi 25.5, on openEuler 22.03 LTS SP4. A different
toolkit or device is not an equivalent reproduction environment.

## Recommended launcher

From the repository checkout, the host needs Python and Docker. The launcher
uses the immutable image digests in [`images.json`](images.json); Docker pulls
the selected image automatically if it is missing. It does not install Python
dependencies on the host. Use an empty **result path**; its parent is created
automatically. Paths are on the Docker host; a remote Docker context needs its
own accessible bind-mount locations.

```bash
# A800: select one idle physical GPU.
python scripts/reproduce_paper.py a800 --device 0 --output results/paper-a800

# 9382: physical device 14 = card 7 / chip 0 on the measured server.
python scripts/reproduce_paper.py ascend9382 --device 14 --card 7 --chip 0 \
  --output results/paper-9382

# CPU-only archived replay; use ascend9382 instead on an arm64 Docker host.
python scripts/reproduce_paper.py a800 --figures-only --output results/paper-figures
```

Add `--plan` to print the exact Docker command without creating directories or
starting Docker. `--suite operators` / `--suite models` selects a complete subset;
`--resume` continues the same experiment and preserves failed attempts. A
figure-only replay requires a fresh directory and cannot use `--resume`.
Alternate Ascend host mount paths can be supplied with `--ascend-root`,
`--npu-smi` and `--ascend-install-info`. The commands below show the equivalent
manual Docker invocation and device prerequisites.

The images retain their published 2026-10-30 source snapshot; subsequent guide
and host-launcher changes in Git do not rewrite that snapshot. The
[validation guide](../docs/validation.md) describes the checked scope.
The full 21-model plan was bundled, not rerun in
its entirety during packaging validation.

## Run on A800

Select an idle device explicitly. The full model suite requires an 80 GB A800
and substantial host RAM for saved model/optimizer snapshots. Docker needs the
NVIDIA Container Toolkit. These commands target a Linux host with the device.

```bash
mkdir -p results
docker pull zhangyi2026/residual-wy-a800:20261030-paper
docker run --rm --gpus 'device=0' --ipc=host \
  -e CUDA_VISIBLE_DEVICES=0 \
  -v "$PWD/results:/results" \
  zhangyi2026/residual-wy-a800:20261030-paper \
  run --output /results/a800
```

This executes the four operators' correctness, extended checks, memory and
complete-backward timing, component controls, and all **17 A800 model cases**.
The three changed-structure model cases remain explicitly marked
`structural_exploration`; their results are not merged with official structures.
Use a single idle GPU throughout; sharing it invalidates performance comparisons.

## Run on Ascend910_9382

The example selects physical device 14 (card 7, chip 0 on the measured dual-chip
server). Check `npu-smi info` and change all three selectors together on another
server. The user must have device access. `--pid=host` lets the existing telemetry
compare process IDs accurately. The measured 25.5 driver required `--privileged`
for device discovery; ordinary explicit device mounts failed to initialize it.
`ASCEND_RT_VISIBLE_DEVICES` still selects the single device used for experiments.

```bash
mkdir -p results
docker pull zhangyi2026/residual-wy-ascend9382:20261030-paper
docker run --rm --privileged --ipc=host --pid=host \
  -v /usr/local/Ascend:/usr/local/Ascend:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -e ASCEND_RT_VISIBLE_DEVICES=14 \
  -e FDR_NPU_SMI_CARD=7 -e FDR_NPU_SMI_CHIP=0 \
  -v "$PWD/results:/results" \
  zhangyi2026/residual-wy-ascend9382:20261030-paper \
  run --output /results/ascend9382
```

Mount the actual `npu-smi` executable path if it differs. Preserve the entire
`/usr/local/Ascend` hierarchy so the toolkit's absolute symlinks resolve. This
executes the four operators, historical component controls and **four full-model
cases**. This is the **9382** artifact, not the separate Ascend950 study.

## Data, plots and resuming

Each successful `run` writes:

- `protocol.json`, `environment.json`, `source-hashes.json` and a source snapshot;
- `jobs/<job>/records.jsonl` or `jobs/<job>/model/`, plus commands and full logs;
- `summary.json`, operator/memory/model CSV tables, and new-measurement PNG/PDF/SVG
  plots under `figures/`.
- `paper-replay/`, with all six original manuscript figures regenerated from the
  separately labelled archived paper data, with their numerical values checked.

The aggregation checks complete runs, all six operator shapes, exact sample
counts, original/optimized pairing and the **B** timing boundary. It never copies
old paper speedups into a new measurement. Recompute/full saving, low-level
operator B and full-model B remain distinct. Original raw records are preserved.
Timing varies with hardware state; exact historical speedups are not guaranteed.

Keep the same image, arguments and output mount, and add `--resume` after a
failure. Completed jobs are skipped; a failed attempt is retained and the retry
gets its own numbered directory. Changed source hashes or job selections are
rejected. An existing output directory without `--resume` is never overwritten.
`--suite operators` or `--suite models` selects a complete subset. For a deliberate
partial validation use `--only <comma-separated-job-IDs>`; it is labelled partial.
List the actual commands without touching a device:

```bash
docker run --rm zhangyi2026/residual-wy-a800:20261030-paper plan
docker run --rm zhangyi2026/residual-wy-a800:20261030-paper verify
```

To reproduce **all six original manuscript figures**, including their original
three-platform comparisons, replay the bundled paper data. This is a CPU-only
operation and is distinct from taking new measurements:

```bash
docker run --rm -v "$PWD/results:/results" \
  zhangyi2026/residual-wy-a800:20261030-paper \
  replay --output /results/paper-replay
```

The same `replay` command works with the 9382 image on an arm64 host, without NPU
devices or CANN mounts. Both check the original numerical figure values. The
A800 plot environment retains NumPy 2.5.4/Matplotlib 3.11.2. Python 3.11 in the
9382 image uses NumPy 1.26.4/Matplotlib 3.10.3 for rendering; rasterization may
differ while numerical figure values must match. Raster bytes can also differ
across operating systems and font-library builds on either platform.
Historical precision controls
retain the paper's **†** marker; model precision differences remain diagnostics,
not proof of training convergence. See [measurement scope](../docs/results.md).

## Building from the measured environment

`build_context.py` verifies the frozen FLA files and creates an artifact archive
from the public source and benchmark directories; `runtime-manifest.json` records every input hash.
The runtime exports contain installed software only, not credentials, caches or
experiment results. No runtime export is committed to Git. Rebuilt images use
the current public source layout and replay `benchmarks/data/`; they do not
replace the immutable October 30 images or their archived source layout.

Prepare `runtime.tar.gz` with relative paths `env/`, `deps/`, `compat/` for A800.
For 9382 use `torch-env/` and `python/` from the recorded Python installation;
also supply `model-deps.tar.gz`, `plot-deps.tar.gz` and
`libkda_residual_native.so`, and `toolchain.tar.gz` containing the measured
openEuler GCC 10.3.1 / binutils and development-header packages at their original
absolute paths (stored relative to `/` in the archive). Package versions and
their bundled license texts are retained, including `/opt/toolchain-packages.txt`.
Include `kernel-headers` as well as `glibc-devel` and `libstdc++-devel`: Triton's
Python/C++ launchers require the Linux/asm headers on an empty cache.
The plot dependencies supplement the runtime without
upgrading its Torch/Triton/NumPy. Preserve venv symlinks. Then:

```bash
python containers/build_context.py --platform a800 \
  --context /path/to/context --fla /path/to/frozen-fla
docker build --platform linux/amd64 -t paper-a800 /path/to/context
# Use --platform ascend9382 and Docker build --platform linux/arm64 for 9382.
```

Both Dockerfiles pin the public base manifest. For offline builds only,
`BASE_IMAGE` can refer to a locally loaded copy after verifying its config and
layer chain against that pinned manifest. Image publication checks that chain
and every uploaded layer; source and runtime hashes are also stored inside the
image at `/opt/runtime-manifest.json`.
