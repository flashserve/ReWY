# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import torch
import triton
import triton.language as tl

from fla.backends import dispatch
from fla.ops.kda.chunk_intra_token_parallel import chunk_kda_fwd_intra_token_parallel
from fla.ops.kda.wy_fast import recompute_w_u_fwd
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.cache import fla_cache_autotune
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.graph import get_static_buffer
from fla.ops.utils.op import exp2, gather, unflatten_program_id
from fla.utils import IS_GATHER_SUPPORTED, IS_NPU, IS_TF32_SUPPORTED, TRITON_ABOVE_3_6_0, autotune_cache_kwargs

if IS_TF32_SUPPORTED:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('tf32')
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr('ieee')
# TF32 rounding makes safe-gate gradient error grow with context length; bf16x3 reduces it and runs faster than TF32.
SAFE_GATE_DOT_PRECISION = tl.constexpr('bf16x3' if TRITON_ABOVE_3_6_0 and not IS_NPU else 'ieee')

################################################################################
# Fused inter + solve_tril kernel: compute off-diagonal Akk and solve in one pass
################################################################################

@triton.heuristics({
    'IS_VARLEN': lambda args: args['cu_seqlens'] is not None,
})
@fla_cache_autotune(
    configs=[
        triton.Config({}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [1, 2, 4, 8]
        for num_stages in [2, 3, 4]
    ],
    key=['BK', 'NC', 'BT', 'HV'],
    restore_value=['db'],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=['B', 'T'])
def chunk_kda_bwd_kernel_intra(
    q,
    k,
    g,
    beta,
    dAqk,
    dAkk,
    dq,
    dq2,
    dk,
    dk2,
    dg,
    dg2,
    db,
    cu_seqlens,
    chunk_indices,
    B,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SAFE_GATE: tl.constexpr,
    GATE_SHIFT: tl.constexpr,
    USE_GATHER: tl.constexpr,
    USE_GRAPH: tl.constexpr = False,
):
    i_kc, i_t = unflatten_program_id(tl.cdiv(K, BK) * NC)
    i_bh = tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)
    i_k, i_i = i_kc // NC, i_kc % NC

    total_tokens = B * T
    if IS_VARLEN:
        i_n = tl.load(chunk_indices + i_t * 2).to(tl.int32)
        if USE_GRAPH and i_n < 0:
            return
        i_t = tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_k = i_k * BK + tl.arange(0, BK)
    m_k = o_k < K

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    beta += bos * HV + i_hv

    dAqk += (bos * HV + i_hv) * BT
    dAkk += (bos * HV + i_hv) * BT
    dq += (bos * HV + i_hv) * K
    dq2 += (bos * HV + i_hv) * K
    dk += (bos * HV + i_hv) * K
    dk2 += (bos * HV + i_hv) * K
    dg += (bos * HV + i_hv) * K
    dg2 += (bos * HV + i_hv) * K
    db += bos * HV + i_hv

    o_i = tl.arange(0, BC)
    o_c = i_ti + o_i
    m_c = o_c < T
    m_ck = m_c[:, None] & m_k[None, :]
    m_dAf = m_c[:, None] & (o_i[None, :] < BT)
    m_dAt = (o_i[:, None] < BT) & m_c[None, :]
    p_g = g + o_c[:, None] * (HV*K) + o_k[None, :]
    b_g = tl.load(p_g, mask=m_ck, other=0.0).to(tl.float32)

    p_b = beta + o_c * HV
    b_b = tl.load(p_b, mask=m_c, other=0.0)

    b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
    b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)
    if i_i > 0:
        # [1, BK]
        b_gn = tl.load(g + i_ti * HV*K + o_k, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(0, i_i):
            o_j = i_t * BT + i_j * BC + o_i
            m_jk = (o_j < T)[:, None] & m_k[None, :]
            p_k = k + o_j[:, None] * (H*K) + o_k[None, :]
            p_gk = g + o_j[:, None] * (HV*K) + o_k[None, :]
            p_dAqk = dAqk + o_c[:, None] * (HV*BT) + (i_j * BC + o_i)[None, :]
            p_dAkk = dAkk + o_c[:, None] * (HV*BT) + (i_j * BC + o_i)[None, :]
            # [BC, BK]
            b_k = tl.load(p_k, mask=m_jk, other=0.0)
            b_gk = tl.load(p_gk, mask=m_jk, other=0.0)
            b_kg = b_k * exp2(b_gn - b_gk)
            # [BC, BC]
            b_dAqk = tl.load(p_dAqk, mask=m_dAf, other=0.0)
            b_dAkk = tl.load(p_dAkk, mask=m_dAf, other=0.0)
            # [BC, BK]
            b_dq2 = tl.dot(b_dAqk, b_kg, b_dq2)
            b_dk2 = tl.dot(b_dAkk, b_kg, b_dk2)
        b_gqn = exp2(b_g - b_gn)
        b_dq2 *= b_gqn
        b_dk2 *= b_gqn

    o_i = tl.arange(0, BC)
    m_dA = (i_ti + o_i) < T
    o_dA = (i_ti + o_i) * HV*BT + i_i * BC
    p_kj = k + i_ti * H*K + o_k
    p_gkj = g + i_ti * HV*K + o_k

    p_q = q + o_c[:, None] * (H*K) + o_k[None, :]
    p_k = k + o_c[:, None] * (H*K) + o_k[None, :]
    b_q = tl.load(p_q, mask=m_ck, other=0.0)
    b_k = tl.load(p_k, mask=m_ck, other=0.0)
    # match the forward anchor when recomputing query-side gate factors
    if SAFE_GATE:
        if USE_GATHER:
            b_gn = gather(b_g, tl.zeros([1, BK], dtype=tl.int16), axis=0)
        else:
            b_gn = tl.load(g + i_ti * HV*K + o_k, mask=m_k, other=0)[None, :]
        b_gn = b_gn - 0.5 * (BC - 1) * GATE_SHIFT

        p_dAqk = dAqk + o_c[:, None] * (HV*BT) + (i_i * BC + o_i)[None, :]
        p_dAkk = dAkk + o_c[:, None] * (HV*BT) + (i_i * BC + o_i)[None, :]
        b_dAqk_diag_qk = tl.load(p_dAqk, mask=m_dAf, other=0.0).to(tl.float32)
        b_dAkk_diag_qk = tl.load(p_dAkk, mask=m_dAf, other=0.0).to(tl.float32)

        m_i_diag_qk = (o_i[:, None] >= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
        m_j_diag_qk = (i_ti + o_i[:, None]) < T

        b_dAqk_diag_qk = tl.where(m_i_diag_qk, b_dAqk_diag_qk, 0.)
        b_dAkk_diag_qk = tl.where(m_i_diag_qk, b_dAkk_diag_qk, 0.)
        b_g_diag_qk = tl.where(m_j_diag_qk, b_g - b_gn, 0.)
        b_gq_diag_qk = tl.where(m_j_diag_qk, exp2(b_g_diag_qk), 0.)
        b_gk_diag_qk = tl.where(m_j_diag_qk, exp2(-b_g_diag_qk), 0.)

        b_k_exp_diag_qk = b_k * b_gk_diag_qk
        b_dq2 += tl.dot(b_dAqk_diag_qk, b_k_exp_diag_qk, input_precision=SAFE_GATE_DOT_PRECISION) * b_gq_diag_qk
        b_dk2 += tl.dot(b_dAkk_diag_qk, b_k_exp_diag_qk, input_precision=SAFE_GATE_DOT_PRECISION) * b_gq_diag_qk
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            # [BC]
            b_dAqk = tl.load(dAqk + o_dA + j, mask=m_dA, other=0)
            b_dAkk = tl.load(dAkk + o_dA + j, mask=m_dA, other=0)
            # [BK]
            b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            m_i = o_i[:, None] >= j
            # [BC, BK]
            b_gqk = exp2(b_g - b_gkj[None, :])
            b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, 0.)
            b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kj[None, :] * b_gqk, 0.)

            p_kj += H*K
            p_gkj += HV*K

    beta_fallback = (tl.abs(b_b) < 1.e-6) & m_c
    b_db = tl.sum(b_dk2 * b_k, 1)
    tl.atomic_add(db + o_c * HV, b_db, mask=beta_fallback, sem="relaxed")
    b_dk2 *= b_b[:, None]

    p_dq = dq + o_c[:, None] * (HV*K) + o_k[None, :]
    p_dq2 = dq2 + o_c[:, None] * (HV*K) + o_k[None, :]

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, mask=m_ck, other=0.0)
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), mask=m_ck)

    tl.debug_barrier()
    b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

    NC = min(NC, tl.cdiv(T - i_t * BT, BC))
    if i_i < NC - 1:
        # [1, BK]
        b_gn = tl.load(g + (min(i_ti + BC, T) - 1) * HV*K + o_k, mask=m_k, other=0).to(tl.float32)[None, :]
        for i_j in range(i_i + 1, NC):
            o_j = i_t * BT + i_j * BC + o_i
            m_j = o_j < T
            m_jk = m_j[:, None] & m_k[None, :]
            m_dAj = (o_i[:, None] < BT) & m_j[None, :]
            p_q = q + o_j[:, None] * (H*K) + o_k[None, :]
            p_k = k + o_j[:, None] * (H*K) + o_k[None, :]
            p_gk = g + o_j[:, None] * (HV*K) + o_k[None, :]
            p_b = beta + o_j * HV
            p_dAqk = dAqk + (i_i * BC + o_i)[:, None] + o_j[None, :] * (HV*BT)
            p_dAkk = dAkk + (i_i * BC + o_i)[:, None] + o_j[None, :] * (HV*BT)
            # [BC]
            b_b = tl.load(p_b, mask=m_j, other=0.0)
            # [BC, BK]
            b_q = tl.load(p_q, mask=m_jk, other=0.0)
            b_kb = tl.load(p_k, mask=m_jk, other=0.0) * b_b[:, None]
            b_gk = tl.load(p_gk, mask=m_jk, other=0.0).to(tl.float32)
            # [BC, BC]
            b_dAqk = tl.load(p_dAqk, mask=m_dAj, other=0.0)
            b_dAkk = tl.load(p_dAkk, mask=m_dAj, other=0.0)

            # [BC, BK]
            b_gkn = exp2(b_gk - b_gn)
            b_qg = b_q * tl.where(m_j[:, None], b_gkn, 0)
            b_kbg = b_kb * tl.where(m_j[:, None], b_gkn, 0)
            # [BC, BK]
            # (SY 09/17) important to not use bf16 here to have a good precision.
            b_dkt = tl.dot(b_dAqk, b_qg, b_dkt)
            b_dkt = tl.dot(b_dAkk, b_kbg, b_dkt)
        b_dkt *= exp2(b_gn - b_g)
    o_dA = i_ti * HV*BT + i_i * BC + o_i
    p_qj = q + i_ti * H*K + o_k
    p_kj = k + i_ti * H*K + o_k
    p_gkj = g + i_ti * HV*K + o_k
    p_bj = beta + i_ti * HV
    # use the same anchor for key-side gradients so the shared offset cancels
    if SAFE_GATE:
        if USE_GATHER:
            b_gn = gather(b_g, tl.zeros([1, BK], dtype=tl.int16), axis=0)
        else:
            b_gn = tl.load(g + i_ti * HV*K + o_k, mask=m_k, other=0).to(tl.float32)[None, :]
        b_gn = b_gn - 0.5 * (BC - 1) * GATE_SHIFT
        p_q = q + o_c[:, None] * (H*K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_ck, other=0.0)
        p_b = beta + o_c * HV
        b_b = tl.load(p_b, mask=m_c, other=0.0)

        p_dAqk = dAqk + (i_i * BC + o_i)[:, None] + o_c[None, :] * (HV*BT)
        p_dAkk = dAkk + (i_i * BC + o_i)[:, None] + o_c[None, :] * (HV*BT)
        b_dAqk_diag_kk = tl.load(p_dAqk, mask=m_dAt, other=0.0).to(tl.float32)
        b_dAkk_diag_kk = tl.load(p_dAkk, mask=m_dAt, other=0.0).to(tl.float32)

        m_i_diag_kk = (o_i[:, None] <= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
        m_j_diag_kk = (i_ti + o_i[:, None]) < T

        b_dAqk_diag_kk = tl.where(m_i_diag_kk, b_dAqk_diag_kk, 0.)
        b_dAkk_diag_kk = tl.where(m_i_diag_kk, b_dAkk_diag_kk, 0.)
        # ensure numerical stability
        b_g_diag_kk = tl.where(m_j_diag_kk, b_g - b_gn, 0.)
        b_gq_diag_kk = tl.where(m_j_diag_kk, exp2(b_g_diag_kk), 0.)
        b_gk_diag_kk = tl.where(m_j_diag_kk, exp2(-b_g_diag_kk), 0.)

        b_q_exp = b_q * b_gq_diag_kk
        b_kb_exp = b_k * b_b[:, None] * b_gq_diag_kk

        b_dkt += tl.dot(b_dAqk_diag_kk, b_q_exp, input_precision=SAFE_GATE_DOT_PRECISION) * b_gk_diag_kk
        b_dkt += tl.dot(b_dAkk_diag_kk, b_kb_exp, input_precision=SAFE_GATE_DOT_PRECISION) * b_gk_diag_kk
    else:
        for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
            # [BC,]
            b_dAqk = tl.load(dAqk + o_dA + j * HV*BT)
            b_dAkk = tl.load(dAkk + o_dA + j * HV*BT)
            # [BK,]
            b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
            b_kbj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32) * tl.load(p_bj)
            b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
            # [BC, BK]
            m_i = o_i[:, None] <= j
            b_gkq = exp2(b_gkj[None, :] - b_g)
            b_dkt += tl.where(m_i, b_dAqk[:, None] * b_qj[None, :] * b_gkq, 0.)
            b_dkt += tl.where(m_i, b_dAkk[:, None] * b_kbj[None, :] * b_gkq, 0.)

            p_qj += H*K
            p_kj += H*K
            p_gkj += HV*K
            p_bj += HV
    p_dk = dk + o_c[:, None] * (HV*K) + o_k[None, :]
    p_dk2 = dk2 + o_c[:, None] * (HV*K) + o_k[None, :]
    p_dg = dg + o_c[:, None] * (HV*K) + o_k[None, :]
    p_dg2 = dg2 + o_c[:, None] * (HV*K) + o_k[None, :]

    b_dg2 += (b_dk2 - b_dkt) * b_k + tl.load(p_dg, mask=m_ck, other=0.0)
    b_dk2 += tl.load(p_dk, mask=m_ck, other=0.0)
    b_dk2 += b_dkt

    tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), mask=m_ck)
    tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), mask=m_ck)

def chunk_kda_bwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    dAqk: torch.Tensor,
    dAkk: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    db: torch.Tensor,
    dg: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.LongTensor | None = None,
    chunk_size: int = 64,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    use_graph: bool = False,
):
    B, T, H, K, HV = *k.shape, g.shape[2]
    BT = chunk_size
    BC = min(16, BT)
    BK = min(32, triton.next_power_of_2(K))

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)
    NK = triton.cdiv(K, BK)

    if use_graph:
        dq2 = get_static_buffer("intra_dq2", tuple(dq.shape), dq.dtype, dq.device)
        dk2 = get_static_buffer("intra_dk2", tuple(dk.shape), dk.dtype, dk.device)
        dg2 = get_static_buffer("intra_dg2", tuple(dg.shape), torch.float, dg.device)
    else:
        dq2 = torch.empty_like(dq)
        dk2 = torch.empty_like(dk)
        dg2 = torch.empty_like(dg, dtype=torch.float)
    grid = (NK * NC * NT, B * HV)
    chunk_kda_bwd_kernel_intra[grid](
        q=q,
        k=k,
        g=g,
        beta=beta,
        dAqk=dAqk,
        dAkk=dAkk,
        dq=dq,
        dq2=dq2,
        dk=dk,
        dk2=dk2,
        dg=dg,
        dg2=dg2,
        db=db,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        B=B,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        BK=BK,
        NC=NC,
        SAFE_GATE=safe_gate,
        GATE_SHIFT=(-lower_bound if lower_bound is not None else 2.) * RCP_LN2,
        USE_GATHER=IS_GATHER_SUPPORTED,
        USE_GRAPH=use_graph,
    )
    dq = dq2
    dk = dk2
    dg = dg2

    return dq, dk, db, dg
