# Release validation — 2026-10-27

## Reading this history

This page preserves the original release record and dated addenda. For current
installation/use, start at [the documentation index](README.md).

**2026-10-30:** [A800 and 9382 paper images](../containers/README.md) were published.
Their immutable digests are pinned in [images.json](../containers/images.json).
Packaging validation included four-operator checks
on both measured environments and an additional 9382 cold-cache Docker check.
All six archived figures were replayed; the 21 model recipes were bundled but
not all rerun during this packaging verification. The original unpublished-image
limitation below describes the earlier library packaging date, not these images.

## Original release record

The following is the original packaging record. The 2026-10-29 Ascend910_9382
matrix-intra correction and full-model performance retest are documented in
[9382 validation](ascend9382.zh-CN.md). Its kernel regression evidence does not
establish full-model numerical/convergence acceptance or a complete new wheel
acceptance. Historical records below remain unchanged.

The v0.1 packaging work did not run new performance experiments and does not
introduce or replace any historical speedup claim.

## Passed

- Python wheel built and installed into a separate target directory on Linux.
  Acceptance imported that installed wheel, not the source checkout.
- The frozen FLA dependency was also built and installed as a wheel; the launcher
  then completed an optimized training update with both packages imported from
  their installation directories, without using the experiment FLA source path.
- NVIDIA A800-SXM4-80GB, torch 2.9.1+cu128, Triton 3.5.1, pinned FLA sources.
- Independent CPU FP64 recurrence: four operators, original and optimized;
  KDA/GDN2 in recomputation and native all-save modes, GDN/DeltaNet in native
  recomputation mode. All 12 configurations passed forward output, final state
  and every input gradient (ordinary relative L2 <= 0.008; gate/beta/vector gate
  <= 0.02; near-zero absolute <= 1e-6). Test shape B=1,T=64,H=2,K=V=128.
- 18 tests passed from the installed wheel: control-plane isolation, nested
  contexts, invalid configuration, no-hit detection, CLI help, four upstream FLA
  integrations, backward after disabling, absent state/noncontiguous input,
  strict rejection and visible fallback, fused KDA model-gate gradients,
  DeltaNet/GDN2 layer execution and a small Qwen3.5 CausalLM backward.
- The original FLA files were not modified. Public autograd graphs retain their
  selected backward after leaving the backend context.
- The launcher executed `examples/train_step.py` with the optimized backend:
  two training updates, two optimized forward/backward submissions, no fallback.

The small model tests establish dispatch and finite backward execution, not
full-model training equivalence, long-run convergence or performance. The
independent reference test is deliberately small and does not certify arbitrary
shapes, dtypes or model configurations.

## Original release limitations (before the 2026-10-29 NPU addendum)

- The existing NPU test host rejected its connection. NPU kernels and native
  extension source are included, but optimized NPU dispatch requires the explicit
  experimental flag. Historical kernel evidence is not a release acceptance.
- Docker recipes have not been built into published container images.
- No full six-shape performance suite, full-model benchmark, distributed test,
  activation-checkpoint test or full-graph compiler test was run for this release.
- Official GDN2 adapter requires separately installed upstream code; the FLA
  GDN2 layer was smoke-tested, not every upstream project's dependency setup.

## Run the tests yourself

```bash
python -m pip install '.[test]'
python -m pytest -q -m 'not accelerator'
FDR_TEST_DEVICE=cuda python -m pytest -q
fdr verify --device cuda
```

Verification and dispatch reports contain the actual runtime counts. Inspect
fallback/rejection counters before treating a run as optimized.

## 2026-10-29 Ascend950 experiment addendum

The explicit 950 runner has now passed four-operator acceptance and the fixed
six-shape combined suite on Ascend950DT_9582, plus standalone KDA/GDN2 residual
and FP32 matrix-intra controls. It installs uniform runtime scheduling
adaptations required for correct execution on this device; ordinary package
import does not install these experimental adaptations. The frozen FLA disk
sources and original precision contract remain intact.

The final v8 run includes 240 full-shape acceptance rows (120 exact repeat
checks), CPU FP64 recurrence checks at T=64, actual-FLA-input matrix regressions,
and 168 paired public-autograd B/FB timing cases. All 20,160 raw observations
are retained. This is separate from earlier release and model validation;
it does not certify arbitrary shapes, FP16/varlen, graphs, or convergence.
See [950 scope, results and reproduction](ascend950.zh-CN.md).
