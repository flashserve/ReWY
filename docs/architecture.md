# Stable interface, replaceable kernels

The public package does not import Torch or modify frameworks on import.
`enable()` installs explicit adapters; `disable()` selects original behavior.
The launcher installs adapters before executing the user's script. Context-local
selection keeps nested `use_backend` scopes independent. Enable in each worker
before model import; automatic distributed-worker injection is not provided.

KDA and GDN use FLA's existing registry. DeltaNet and GDN2 lack matching public
dispatch hooks in the frozen baseline, so their public module aliases are bound
once during installation. References imported before installation may still
point at the original functions: initialize first. Qwen and official GDN2 have
separate explicit model bindings. No scanning or rewriting of arbitrary modules,
`sitecustomize`, `.pth` injection or source-tree modification is performed.

An optimized forward uses a private copy of the frozen public autograd contract.
Its own autograd Function permanently selects its backward, including when the
user exits a context or disables the backend before backward. KDA and GDN2
backward orchestration functions are cloned with private global bindings; their
original kernel modules are never patched. GDN and DeltaNet use a dedicated
complete-backward adapter. The original forward, precision and save semantics
are preserved. Original calls use the frozen upstream autograd classes.

## Updating kernels

1. Change the device kernel or its private orchestration binding.
2. Preserve public inputs/outputs, state layout, supported gradients, precision
   and save policy. A change to saved tensors requires a matched forward/backward.
3. Record source changes and validation. Run CPU control-plane tests, wheel
   installation tests and accelerator acceptance on affected devices.
4. Test an actual FLA/model call and check optimized-backward counters. A passing
   standalone kernel test does not prove that model dispatch works.
5. Release a new version; keep prior tags and dependency locks for reproduction.

Changing the baseline requires reviewing the compatibility modules and updating
source fingerprints deliberately. Do not bypass the baseline check to accept an
unreviewed FLA revision. The pinned source contract is an initial-release tradeoff
that makes the boundary auditable; later supported revisions can have separate
adapters.

## Precision and scheduling

CUDA KDA uses residual WY plus FP32 endpoint-scaled matrix intra, with separate
scheduling. CUDA GDN2 uses the FP32-repaired joint consumer. CUDA scalar operators
use the A800 liveness consumer. There is no per-benchmark-case dispatch or timing
autotuner choosing between original and optimized implementations.

Ascend kernels retain their hardware-wide schedules. Existing FP32 matrices,
local accumulation and values used by existing gradients must remain FP32.
New algebraic intermediates follow the documented experiment contract; they are
not permission to quantize an old matrix under a new name. Numerical stability
branches within a kernel are different from unsupported-interface fallback.
