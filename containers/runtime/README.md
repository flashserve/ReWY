# Build a current-library image

These recipes install the current checkout and expose the `fdr` CLI. For the
published manuscript environments and one-command experiments, use the
[paper images](../README.md) instead.

Run builds from the repository root:

```bash
docker build -f containers/runtime/Dockerfile.cuda -t fast-delta-rule:cuda .
docker run --rm --gpus 'device=0' fast-delta-rule:cuda verify --device cuda

# Mount your training project so its relative imports/data paths keep working.
docker run --rm --gpus 'device=0' --ipc=host \
  -v "$PWD:/workspace" -w /workspace fast-delta-rule:cuda \
  run --backend optimized --report results/dispatch.json examples/train_step.py
```

For NPU, supply an existing matching CANN/TorchNPU/Triton-Ascend base:

```bash
docker build -f containers/runtime/Dockerfile.npu \
  --build-arg NPU_BASE=your-validated-ascend-image -t fast-delta-rule:npu .
```

Configure host device access for that environment and enable
`FDR_EXPERIMENTAL_NPU=1` when using the experimental public backend.
[Compatibility](../../docs/compatibility.md) defines its input/validation scope.
The published 9382 paper image's tested mounts are documented separately.

These build recipes moved here from the repository root. Build context remains
the repository root (`.`); dependencies and entry points are unchanged.
