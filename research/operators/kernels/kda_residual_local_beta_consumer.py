# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Experimental residual-WY adaptation; upstream attribution retained.
import triton
import triton.language as tl
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.op import exp2
from fla.utils import IS_NVIDIA_HOPPER, IS_NVIDIA_SM100, autotune_cache_kwargs, check_shared_mem

BK_LIST = [32, 64] if check_shared_mem() else [16, 32]
BV_LIST = [64, 128] if check_shared_mem('ampere') else [16, 32]
NUM_WARPS = [2, 4] if IS_NVIDIA_HOPPER else [2, 4, 8]

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@triton.jit(do_not_specialize=['T'])
def residual_wy_local_beta_consumer(
    q,
    k,
    v,
    v_new,
    g,
    beta,
    A,
    h,
    do,
    dh,
    dq,
    dk,
    dv,
    dv2,
    dg,
    db,
    dA,
    db_v,
    cu_seqlens,
    chunk_indices,
    scale,
    T,
    TOTAL_TOKENS: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    STATE_V_FIRST: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GRAPH: tl.constexpr = False,
):
    i_t, i_bh = (tl.program_id(0) // tl.cdiv(K, BK)).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_tg = i_t.to(tl.int64)
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        if USE_GRAPH and i_n < 0:
            return
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        T = (eos - bos).to(tl.int32)
        NT = tl.cdiv(T, BT)
    else:
        NT = tl.cdiv(T, BT)
        i_tg = (i_b * NT + i_t).to(tl.int64)
        bos, eos = (i_b * T).to(tl.int64), (i_b * T + T).to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BT)
    m_t = o_t < T
    m_last = (o_t == min(T, i_t * BT + BT) - 1)

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    v += (bos * HV + i_hv) * V
    v_new += (bos * HV + i_hv) * V
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv
    A += (bos * HV + i_hv) * BT
    h += (i_tg * HV + i_hv) * K*V
    do += (bos * HV + i_hv) * V
    dh += (i_tg * HV + i_hv) * K*V
    dq += (bos * HV + i_hv) * K
    dk += (bos * HV + i_hv) * K
    dv += (bos * HV + i_hv) * V
    dv2 += (bos * HV + i_hv) * V
    dg += (bos * HV + i_hv) * K
    db += bos * HV + i_hv
    dA += (bos * HV + i_hv) * BT

    p_beta = beta + o_t * HV
    b_beta = tl.load(p_beta, mask=m_t, other=0.0)
    fallback = (tl.abs(b_beta) < 1.e-6) & m_t

    b_db = tl.zeros([BT], dtype=tl.float32)

    i_k = (tl.program_id(0) % tl.cdiv(K, BK))
    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K
    m_tk = m_t[:, None] & m_k[None, :]

    p_k = k + o_t[:, None] * (H*K) + o_k[None, :]
    p_g = g + o_t[:, None] * (HV*K) + o_k[None, :]
    b_k = tl.load(p_k, mask=m_tk, other=0.0)
    b_g = tl.load(p_g, mask=m_tk, other=0.0).to(tl.float32)

    p_gn = g + (min(T, i_t * BT + BT) - 1).to(tl.int64) * HV*K + o_k
    b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)

    b_dq = tl.zeros([BT, BK], dtype=tl.float32)
    b_dk = tl.zeros([BT, BK], dtype=tl.float32)
    b_dw = tl.zeros([BT, BK], dtype=tl.float32)
    b_dgk = tl.zeros([BK], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_tv = m_t[:, None] & (o_v[None, :] < V)
        m_h = (o_v[:, None] < V) & m_k[None, :]
        p_v_new = v_new + o_t[:, None] * (HV*V) + o_v[None, :]
        p_do = do + o_t[:, None] * (HV*V) + o_v[None, :]
        if STATE_V_FIRST:
            p_h = h + o_v[:, None] * K + o_k[None, :]
            p_dh = dh + o_v[:, None] * K + o_k[None, :]
        else:
            p_h = h + o_v[:, None] + o_k[None, :] * V
            p_dh = dh + o_v[:, None] + o_k[None, :] * V
        p_dv = dv + o_t[:, None] * (HV*V) + o_v[None, :]
        # [BT, BV]
        b_v_new = tl.load(p_v_new, mask=m_tv, other=0.0)
        b_do = tl.load(p_do, mask=m_tv, other=0.0)
        # [BV, BK]
        b_h = tl.load(p_h, mask=m_h, other=0.0)
        b_dh = tl.load(p_dh, mask=m_h, other=0.0)
        # [BT, BV]
        b_dv = tl.load(p_dv, mask=m_tv, other=0.0)

        b_dgk += tl.sum(b_h * b_dh, axis=0)
        b_dq = tl.dot(b_do, b_h.to(b_do.dtype), b_dq)
        b_dk = tl.dot(b_v_new, b_dh.to(b_v_new.dtype), b_dk)
        # P = A^T dR; use the same P for both residual VJPs.
        b_dvb = b_dv  # P was computed once by the producer.
        b_dw = tl.dot(b_dvb.to(b_v_new.dtype), b_h.to(b_v_new.dtype), b_dw)
        tl.debug_barrier()  # DO NOT REMOVE THIS LINE!

    b_gk_exp = exp2(b_g)
    b_gb = b_gk_exp * b_beta[:, None]
    b_dgk *= exp2(b_gn)
    b_dq = b_dq * b_gk_exp * scale
    b_dk = b_dk * tl.where(m_t[:, None], exp2(b_gn[None, :] - b_g), 0)

    b_kg = b_k * b_gk_exp

    # Direct VJP for E = beta * exp(g) * k: -P h^T.
    b_dkgb = -b_dw
    if tl.sum(fallback.to(tl.int32), 0) > 0:
        b_db += tl.sum(b_dkgb * b_kg, 1)

    p_q = q + o_t[:, None] * (H*K) + o_k[None, :]
    b_q = tl.load(p_q, mask=m_tk, other=0.0)
    b_kdk = b_k * b_dk
    b_dgk += tl.sum(b_kdk, axis=0)
    b_dg = b_q * b_dq - b_kdk + m_last[:, None] * b_dgk + b_kg * b_dkgb * b_beta[:, None]
    b_dk = b_dk + b_dkgb * b_gb

    p_dq = dq + o_t[:, None] * (HV*K) + o_k[None, :]
    p_dk = dk + o_t[:, None] * (HV*K) + o_k[None, :]
    p_dg = dg + o_t[:, None] * (HV*K) + o_k[None, :]
    tl.store(p_dq, b_dq.to(p_dq.dtype.element_ty), mask=m_tk)
    tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), mask=m_tk)
    tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), mask=m_tk)

    if tl.sum(fallback.to(tl.int32), 0) > 0:
        tl.atomic_add(db + o_t * HV, b_db, mask=fallback, sem="relaxed")
