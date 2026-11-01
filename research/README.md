# Operator and Model Reproduction

[Documentation](../docs/README.md) · [Docker experiments](../containers/README.md) ·
[Figure data](../benchmarks/README.md)

The public API lives in `../src/fast_delta_rule/`. This directory keeps the
historical operator/model implementations required to rerun the reported
experiments and paired ablations. Dated names distinguish measured versions,
not installation steps.

| Study | Source |
| --- | --- |
| Residual WY, unified adjoints and Gram elimination | `operators/kernels/`, `operators/delta_algebra_20261024/`, `operators/family_completion_20261024/` |
| RTX scalar and vector schedules | `operators/gpu_target120_20261025/`, `operators/gpu_vector_followup_20261026/` |
| KDA matrix intra and allocation controls | `operators/gpu_matrix_20261024/`, `operators/family_completion_20261024/space_audit_20261026.py` |
| Historical 9382 native matrix control | `operators/cannbot_npu_20261024/` |
| A800 scalar and vector ablations | `operators/a800_scalar/`, `operators/a800_ablation/` |
| A800 model backward | `models/a800_shared/`, `models/a800_kda_*`, `models/a800_fla_*` |
| 9382 model backward | `models/ascend9382/`, `models/ascend9382_scalar/` |
| Corrected 9382/950 matrix intra and 950 preflight | `../src/fast_delta_rule/kernels/`, `../scripts/npu950_*.py` |
| H100/H200 operator and model experiments | [Pinned statefix1 image](../containers/hopper-statefix/README.zh-CN.md) |

## Inspect a Plan

These commands do not import accelerator modules or run measurements:

```bash
python research/run_model.py --list
python research/run_model.py --case b1t2048gpu0r2
python research/run_operator.py --study a800-scalar --op gdn --mode bench
python research/run_operator.py --study rtx-components --op gdn --mode audit
```

`model_runs.json` lists all 21 A800/9382 configurations, source groups,
parameters and timing boundaries. It contains no internal archive paths.
Add `--run --output PATH` only in the matching environment with an explicitly
selected idle device. Outputs must use a new path. Set `CUDA_VISIBLE_DEVICES`
or `ASCEND_RT_VISIBLE_DEVICES`; for NPU telemetry also set `FDR_NPU_SMI_CARD`
and `FDR_NPU_SMI_CHIP` to the actual physical mapping.

Operator launchers support correctness, extended checks, timing and allocation
audits. Matched R/A controls and six-second extra warmup in the A800/scalar
timing scripts are preserved. Model scripts retain AdamW updates and ABBA order;
reported model gains use only the complete backward interval.

## Historical Controls

Lower-precision matrix controls are retained only for reproducing the labelled
historical comparisons, not as current default kernels or strict-precision
claims. [Results and qualifications](../docs/results.md) distinguish them.

The native historical 9382 control can be built with:

```bash
bash research/operators/cannbot_npu_20261024/native/build.sh
```

Use `KDA_PYTHON` to select its Python environment. It needs the matching CANN
toolchain; no prebuilt `.so` is distributed. Current public 9382/950 paths have
no native-library dependency. See the [950 procedure](../docs/ascend950.zh-CN.md)
for its separate timing boundary and required preflight.

Internal archives, logs, unused deployment workers and source-origin ledgers
are not part of this source distribution. Published Docker images preserve
their own recorded environments. New measurements must retain their raw paired
samples, precision policy and actual runtime source identity in a new output
directory.
