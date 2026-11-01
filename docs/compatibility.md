# Compatibility

## Initial release envelope

- Linux; FLA revision `a59bc5df8c5841dce2169bb00e4027287197ba77`.
- BF16 q/k/v, `[B,T,H,D]`, K=V=128, equal query/key/value head counts.
- Chunk size 64, T a positive multiple of 64; B and H are not benchmark case IDs.
- Fixed-length batches. Packed sequences, context parallelism, higher-order
  gradients and static varlen graph buffers are outside the validated envelope.
- Default scale `1/sqrt(K)`. Original state and output layouts are preserved.
- KDA/GDN2: recomputation or their native `disable_recompute=True` save policy.
- GDN/DeltaNet: upstream public save semantics; no invented all-save argument.
- KDA fused gate/beta and Q/K normalization are accepted on CUDA. GDN/GDN2 fused
  gate input and safe-gate transforms are rejected pending separate validation.
- Full-graph `torch.compile`, DDP/FSDP, activation checkpointing, concurrent host
  threads and complete-model convergence are not claimed as release-validated.
  CUDA graph replay requires its own test; Python counters do not count replays.

`--fallback error` is the default. `--fallback original` explicitly permits
unsupported calls to run the original implementation and emits a warning plus
reason counters. Kernel compilation/runtime failures are never swallowed as
compatibility fallback. No-fallback is required for benchmark claims.

## CUDA

The acceptance environment is A800-SXM4-80GB, torch 2.9.1+cu128 and Triton 3.5.1.
Other GPUs need their own acceptance; historical laptop measurements do not
prove this release is faster on every GPU. First use includes Triton compilation.
Warm up before timing. Do not interpret kernel speedups as full-model speedups.

H100/H200 **research** results are available in the [statefix1 results](results.md#h100-and-h200-operators), using Torch 2.12.1+cu126 / Triton 3.7.1. Those measured research schedules and embedded checks do not extend the public-library release acceptance envelope above.

## Ascend

Target: Ascend910_9382, with matching installed CANN / torch_npu / Triton-Ascend.
The historical native extension uses the CANN `dav-2201` compiler target;
that toolchain name does not change the reported physical device model.

**Experimental packaging:** optimized NPU dispatch remains gated by
`FDR_EXPERIMENTAL_NPU=1`. The 2026-10-29 9382 correction has real-input and
tail/gate regression evidence; it does not establish full-model numerical or
training-convergence acceptance. See [9382 validation](ascend9382.zh-CN.md).

The current NPU interface additionally requires explicit initial state and final
state output. All current 9382 paths use Triton-Ascend. KDA/GDN2 select fixed
32x64 FP32 matrix tiles within the existing K=V=128 release envelope. The old
native extension and BF16-coefficient GDN2 implementation remain historical
sources, are not selected, and are not strict-precision results. No native
workspace or `FDR_NPU_LIBRARY` is needed by the corrected dispatch.
Use `fdr verify --device npu` before experiments.

## Ascend950 experimental validation (2026-10-29)

Physical Ascend950 devices select a separate experimental Triton-Ascend path.
KDA no longer requires the dav-2201 native extension there; KDA/GDN2 use a new
FP32 matrix intra with separately scheduled residual WY. GDN/DeltaNet retain
their NPU candidate algorithms. All four passed the fixed six-shape public
forward/backward suite on Ascend950DT_9582; KDA/GDN2 standalone residual and
matrix-intra benchmarks also passed. The dedicated `npu950_experiment.py`
preflight installs required uniform UB and task-scheduling adaptations.
Use `scripts/install.sh npu950`, not the legacy native build. Confirm device
availability before running, and pass the expected source manifest. See
[950 protocol and limitations](ascend950.zh-CN.md) for the explicit runner,
ablation boundaries and the historical GDN2 NPU precision correction.

## Model integrations

`--integration qwen3_5` binds the documented Transformers Qwen3.5 delta-rule
entry. `--integration gdn2` binds `lit_gpt.gdn2.chunk_gdn2`, requiring the official
GDN2 project on Python's path. Both original and optimized selections use the
same FLA core through these adapters. This is not a comparison against each
model repository's unrelated default backend.

An interface adapter does not certify every configuration of a model family.
Inspect the runtime counters; an optimized run with no optimized backward is an
error unless forward-only operation was explicitly requested.
