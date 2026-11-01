"""950 entry for the corrected FP32 matrix-intra dataflow.

Reuse the exact C32/K64 kernels validated on 9382: valid-lane loads followed
by arithmetic masks, bounded diagonal factors, scalar-beta partial reduction
and direct vector-beta output. Core counts come from the current device.
This is a conservative starting schedule, not 950 hardware acceptance/tuning.
The physical-device guard is independent of the shared source filenames.
"""

SCHEDULE_REVISION = 'npu950-fp32-intra-c32k64-20261029'


def _validate(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, *, vector_beta,
              cu_seqlens=None, chunk_indices=None, chunk_size=64,
              safe_gate=False, lower_bound=None, use_graph=False):
    import torch
    from fast_delta_rule.backends.npu_target import require_950
    require_950(k.device)
    if cu_seqlens is not None or chunk_indices is not None or chunk_size != 64 or safe_gate or use_graph:
        raise ValueError('950 matrix intra supports fixed sequences, C64, and post-activation gates')
    tensors = (q, k, g, beta, dAqk, dAkk, dq, dk, db, dg)
    if any(x.device != k.device or not x.is_contiguous() for x in tensors):
        raise ValueError('950 matrix intra requires contiguous tensors on one NPU')
    B, T, H, K = k.shape
    if K != 128 or T < 64 or T % 64 or q.shape != k.shape or g.shape != k.shape:
        raise ValueError('950 matrix intra requires equal heads, K128 and complete C64 chunks')
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise ValueError('950 matrix intra requires BF16 q/k')
    if beta.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError('950 matrix intra preserves BF16 or FP32 beta; no implicit downcast')
    if any(x.dtype != torch.float32 for x in (g, dAqk, dAkk, dq, dk, db, dg)):
        raise ValueError('950 matrix intra requires FP32 gates, coefficients and partial gradients')
    if beta.shape != (k.shape if vector_beta else k.shape[:-1]):
        raise ValueError('Unexpected erase/beta shape')
    if dAqk.shape != (B, T, H, 64) or dAkk.shape != dAqk.shape:
        raise ValueError('Unexpected FP32 intra coefficient shape')
    if any(x.shape != k.shape for x in (dq, dk, dg)) or db.shape != beta.shape:
        raise ValueError('Unexpected partial-gradient shape')


def kda_intra(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, **kwargs):
    _validate(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, vector_beta=False, **kwargs)
    from .kda_intra_npu9382 import kda_intra as corrected
    return corrected(q, k, g, beta, dAqk, dAkk, dq, dk, db, dg, **kwargs)


def gdn2_intra(q, k, g, b, dAqk, dAkk, dq, dk, db, dg, **kwargs):
    _validate(q, k, g, b, dAqk, dAkk, dq, dk, db, dg, vector_beta=True, **kwargs)
    from .gdn2_intra_npu9382 import gdn2_intra as corrected
    # The common validator accepts the public flag; false is already enforced.
    kwargs.pop('use_graph', None)
    return corrected(q, k, g, b, dAqk, dAkk, dq, dk, db, dg, **kwargs)
