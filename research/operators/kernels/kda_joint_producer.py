"""Upper-triangular P solve product and lower-triangular residual product by tiles."""
import torch
import triton
import triton.language as tl
from .kda_residual_split_consumer import residual_wy_consumer


@triton.jit(do_not_specialize=['T'])
def residual_joint_producer(v, r, a, dr, beta, p, dv, db_v, dt,
                          cu_seqlens, chunk_indices, T,
                          HV: tl.constexpr, V: tl.constexpr, VP: tl.constexpr,
                          BT: tl.constexpr, BR: tl.constexpr,
                          IS_VARLEN: tl.constexpr, USE_GRAPH: tl.constexpr,
                          ROW_FAST: tl.constexpr = False):
    it = tl.program_id(0).to(tl.int64)
    ibh = tl.program_id(1).to(tl.int64)
    ir = tl.program_id(2)
    if ROW_FAST:
        ir = tl.program_id(0) % (BT // BR)
        it = it // (BT // BR)
    ib, ih = ibh // HV, ibh % HV
    if IS_VARLEN:
        seq = tl.load(chunk_indices + 2 * it).to(tl.int32)
        if USE_GRAPH and seq < 0:
            return
        it = tl.load(chunk_indices + 2 * it + 1).to(tl.int64)
        bos = tl.load(cu_seqlens + seq).to(tl.int64)
        T = (tl.load(cu_seqlens + seq + 1) - bos).to(tl.int32)
    else:
        bos = ib * T
    rr = ir * BR + tl.arange(0, BR)
    tt = it * BT + rr
    vv = tl.arange(0, VP)
    valid = tt < T
    off = (bos + tt) * HV + ih
    p_acc = tl.zeros((BR, VP), tl.float32)
    # A^T is upper triangular: preceding reduction blocks are exactly zero.
    for j in range(ir, BT // BR):
        jj = j * BR + tl.arange(0, BR)
        tj = it * BT + jj
        aj = (bos + tj) * HV + ih
        aa = tl.load(a + aj[None, :] * BT + rr[:, None],
                     mask=(tj[None, :] < T) & valid[:, None], other=0.0)
        dd = tl.load(dr + aj[:, None] * V + vv[None, :],
                     mask=(tj[:, None] < T) & (vv[None, :] < V), other=0.0)
        p_acc = tl.dot(aa, dd, p_acc)
    mask = valid[:, None] & (vv[None, :] < V)
    ix = off[:, None] * V + vv[None, :]
    p_dot = p_acc.to(r.dtype.element_ty)
    bb = tl.load(beta + off, mask=valid, other=0.0)
    tl.store(p + ix, p_dot, mask=mask)
    tl.store(dv + ix, (p_acc * bb[:, None]).to(dv.dtype.element_ty), mask=mask)
    fallback = (tl.abs(bb) < 1.e-6) & valid
    r_row = tl.load(r + ix, mask=mask, other=0.0)
    safe_beta = tl.where(fallback | ~valid, 1.0, bb).to(tl.float32)
    dbeta = tl.sum(p_acc * r_row, 1) / safe_beta
    if tl.sum(fallback.to(tl.int32), 0) > 0:
        value = tl.load(v + ix, mask=mask, other=0.0)
        dbeta = tl.where(fallback, tl.sum(p_acc * value, 1), dbeta)
    tl.store(db_v + off, dbeta, mask=valid)
