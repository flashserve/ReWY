"""Independent whole-chunk KDA intra VJP, with unchanged forward/gate semantics.

The algebra also appears in ../kernels/kda_joint_tile.py. This implementation
exposes both old-WY and complete-beta residual-WY interfaces and diagnostics.
It is a GPU implementation, not a translation of AscendC launch/resource sizes.
"""
import torch
import triton
import triton.language as tl
from fla.ops.utils import prepare_chunk_indices


@triton.jit(do_not_specialize=['T'])
def matrix_kernel(q, k, g, beta, L, D, dq, dk, dg, db, flags, cu, ci, T,
                  H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                  BT: tl.constexpr, BK: tl.constexpr, VARLEN: tl.constexpr,
                  BETA_COMPLETE: tl.constexpr, LOWP: tl.constexpr):
    nk = tl.cdiv(K, BK)
    task = tl.program_id(0)
    chunk, ik = task // nk, task % nk
    bh = tl.program_id(1)
    hv = bh % HV
    h = hv // (HV // H)
    if VARLEN:
        seq = tl.load(ci + 2 * chunk).to(tl.int32)
        chunk = tl.load(ci + 2 * chunk + 1).to(tl.int32)
        bos = tl.load(cu + seq).to(tl.int64)
        length = tl.load(cu + seq + 1).to(tl.int64) - bos
    else:
        bos = (bh // HV).to(tl.int64) * T
        length = T
    i = tl.arange(0, BT)
    t = chunk * BT + i
    f = ik * BK + tl.arange(0, BK)
    valid = (t[:, None] < length) & (f[None, :] < K)
    qoff = ((bos + t[:, None]) * H + h) * K + f[None, :]
    off = ((bos + t[:, None]) * HV + hv) * K + f[None, :]
    qv = tl.load(q + qoff, valid, 0).to(tl.float32)
    kv = tl.load(k + qoff, valid, 0).to(tl.float32)
    gv = tl.load(g + off, valid, 0).to(tl.float32)
    bv = tl.load(beta + (bos + t) * HV + hv, t < length, 0).to(tl.float32)
    # A per-feature center cancels exactly in the real-arithmetic identity.
    # Span guard is evaluated on actual cumulative gates, not a changed gate law.
    lo = tl.min(tl.where(t[:, None] < length, gv, float('inf')), 0)
    hi = tl.max(tl.where(t[:, None] < length, gv, -float('inf')), 0)
    matrix_ok = tl.max(hi - lo, 0) < 80.
    if matrix_ok:
        # Do not retain full coefficient matrices across the fallback branch.
        # The fallback streams a row/column at a time instead of reducing a
        # register-resident CxC matrix on every pairwise iteration.
        aoff = ((bos + t[:, None]) * HV + hv) * BT + i[None, :]
        pair = (t[:, None] < length) & (t[None, :] < length)
        lv = tl.load(L + aoff, pair & (i[:, None] >= i[None, :]), 0).to(tl.float32)
        dv = tl.load(D + aoff, pair & (i[:, None] > i[None, :]), 0).to(tl.float32)
        center = (lo + hi) * .5
        plus = tl.where(valid, tl.exp2(gv - center[None, :]), 0.)
        minus = tl.where(valid, tl.exp2(center[None, :] - gv), 0.)
        km, qp, ep = kv * minus, qv * plus, kv * bv[:, None] * plus
        if LOWP:
            a, d = lv.to(tl.bfloat16), dv.to(tl.bfloat16)
            km, qp, ep = km.to(tl.bfloat16), qp.to(tl.bfloat16), ep.to(tl.bfloat16)
        else:
            a, d = lv, dv
        uq = tl.dot(a, km, input_precision='tf32x3') * plus
        ue = tl.dot(d, km, input_precision='tf32x3') * plus
        uk = (tl.dot(tl.trans(a), qp, input_precision='tf32x3') +
              tl.dot(tl.trans(d), ep, input_precision='tf32x3')) * minus
    else:
        uq = tl.zeros((BT, BK), tl.float32)
        ue = tl.zeros((BT, BK), tl.float32)
        uk = tl.zeros((BT, BK), tl.float32)
        for j in range(0, min(BT, length - chunk * BT)):
            source = bos + chunk * BT + j
            kj = tl.load(k + (source * H + h) * K + f, f < K, 0).to(tl.float32)
            qj = tl.load(q + (source * H + h) * K + f, f < K, 0).to(tl.float32)
            gj = tl.load(g + (source * HV + hv) * K + f, f < K, 0).to(tl.float32)
            bj = tl.load(beta + source * HV + hv).to(tl.float32)
            col = ((bos + t) * HV + hv) * BT + j
            row = (source * HV + hv) * BT + i
            lc = tl.load(L + col, (t < length) & (i >= j), 0).to(tl.float32)
            dc = tl.load(D + col, (t < length) & (i > j), 0).to(tl.float32)
            lr = tl.load(L + row, (t < length) & (i <= j), 0).to(tl.float32)
            dr = tl.load(D + row, (t < length) & (i < j), 0).to(tl.float32)
            # Mask exponents before evaluation, avoiding 0 * inf on acausal pairs.
            past = tl.exp2(tl.where(valid & (i[:, None] >= j), gv - gj[None, :], 0.))
            future = tl.exp2(tl.where(valid & (i[:, None] <= j), gj[None, :] - gv, 0.))
            uq += lc[:, None] * kj[None, :] * past
            ue += dc[:, None] * kj[None, :] * past
            uk += (lr[:, None] * qj[None, :] + dr[:, None] * kj[None, :] * bj) * future
    tl.store(flags + bh * tl.num_programs(0) + task, ~matrix_ok)
    bm = t < length
    if BETA_COMPLETE:
        bm &= tl.abs(bv) < 1.e-6
    # WY has initialized db. Each feature tile contributes once; no extra save.
    tl.atomic_add(db + (bos + t) * HV + hv, tl.sum(ue * kv, 1), bm, sem='relaxed')
    oldq = tl.load(dq + off, valid, 0).to(tl.float32)
    oldk = tl.load(dk + off, valid, 0).to(tl.float32)
    oldg = tl.load(dg + off, valid, 0).to(tl.float32)
    eb = ue * bv[:, None]
    tl.store(dq + off, oldq + uq, valid)
    tl.store(dk + off, oldk + eb + uk, valid)
    tl.store(dg + off, oldg + qv * uq + (eb - uk) * kv, valid)


class MatrixIntra:
    def __init__(self, original, residual_original, beta_complete=False, bk=32, warps=4, lowp=True):
        self.original = residual_original if beta_complete else original
        self.beta_complete, self.bk, self.warps, self.lowp = beta_complete, bk, warps, lowp
        self.flags = None
        self.kernel = None
        self.calls = self.fallback_calls = 0

    def __call__(self, q, k, g, beta, dAqk, dAkk, dq, dk, db, dg,
                 cu_seqlens=None, chunk_indices=None, chunk_size=64,
                 safe_gate=False, lower_bound=None, use_graph=False):
        self.calls += 1
        B, T, H, K = k.shape
        HV = g.shape[2]
        args = dict(q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk,
                    dq=dq, dk=dk, db=db, dg=dg, cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices, chunk_size=chunk_size,
                    safe_gate=safe_gate, lower_bound=lower_bound, use_graph=use_graph)
        if (use_graph or chunk_size not in (32, 64) or safe_gate or
                not all(x.is_contiguous() for x in (q, k, g, beta, dAqk, dAkk, dq, dk, db, dg))):
            self.fallback_calls += 1
            self.flags = None
            return self.original(**args)
        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
        nt = triton.cdiv(T, chunk_size) if chunk_indices is None else len(chunk_indices)
        grid = (nt * triton.cdiv(K, self.bk), B * HV)
        # Keep each diagnostic buffer alive through capture/replay via the harness.
        self.flags = torch.empty(grid, device=q.device, dtype=torch.int32)
        self.kernel = matrix_kernel[grid](
            q, k, g, beta, dAqk, dAkk, dq, dk, dg, db, self.flags, cu_seqlens, chunk_indices, T,
            H=H, HV=HV, K=K, BT=chunk_size, BK=self.bk, VARLEN=cu_seqlens is not None,
            BETA_COMPLETE=self.beta_complete, LOWP=self.lowp and k.dtype == torch.bfloat16,
            num_warps=self.warps, num_stages=2)
        return dq, dk, db, dg
