# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

# Adapted: endpoint anchors off diagonal, adaptive centered matrix diagonal,
# FP32 in-place outputs and beta atomic accumulation. Frozen baseline unchanged.
import torch
import triton
import triton.language as tl
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2, unflatten_program_id

@triton.jit(do_not_specialize=['B', 'T'])
def endpoint_matrix_kernel(
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
    flags,
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
    BETA_COMPLETE: tl.constexpr,
    GATE_SHIFT: tl.constexpr,
    USE_GATHER: tl.constexpr,
    USE_GRAPH: tl.constexpr = False,
):
    tl.store(flags + tl.program_id(1) * tl.num_programs(0) + tl.program_id(0), 0)
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
    lo = tl.min(tl.where(m_c[:, None], b_g, float('inf')), 0)
    hi = tl.max(tl.where(m_c[:, None], b_g, -float('inf')), 0)
    center = (lo + hi) * .5
    matrix_ok = tl.max(hi - lo, 0) < 80.
    tl.store(flags + i_bh * tl.num_programs(0) + tl.program_id(0), ~matrix_ok)


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
            b_dq2 = tl.dot(b_dAqk, b_kg, b_dq2, input_precision='tf32x3')
            b_dk2 = tl.dot(b_dAkk, b_kg, b_dk2, input_precision='tf32x3')
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
    if matrix_ok:
        b_gn = center[None, :]

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
        b_dq2 += tl.dot(b_dAqk_diag_qk, b_k_exp_diag_qk, input_precision='tf32x3') * b_gq_diag_qk
        b_dk2 += tl.dot(b_dAkk_diag_qk, b_k_exp_diag_qk, input_precision='tf32x3') * b_gq_diag_qk
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
            b_gqk = exp2(tl.where(m_ck & m_i, b_g - b_gkj[None, :], 0.))
            b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, 0.)
            b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kj[None, :] * b_gqk, 0.)

            p_kj += H*K
            p_gkj += HV*K

    b_db = tl.sum(b_dk2 * b_k, 1)
    b_dk2 *= b_b[:, None]

    p_dq = dq + o_c[:, None] * (HV*K) + o_k[None, :]
    p_dq2 = dq2 + o_c[:, None] * (HV*K) + o_k[None, :]
    p_db = db + o_c * HV

    b_dg2 = b_q * b_dq2
    b_dq2 = b_dq2 + tl.load(p_dq, mask=m_ck, other=0.0)
    tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), mask=m_ck)
    beta_mask = m_c
    if BETA_COMPLETE:
        beta_mask &= tl.abs(b_b) < 1.e-6
    tl.atomic_add(p_db, b_db, mask=beta_mask, sem='relaxed')

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
            b_dkt = tl.dot(b_dAqk, b_qg, b_dkt, input_precision='tf32x3')
            b_dkt = tl.dot(b_dAkk, b_kbg, b_dkt, input_precision='tf32x3')
        b_dkt *= exp2(b_gn - b_g)
    o_dA = i_ti * HV*BT + i_i * BC + o_i
    p_qj = q + i_ti * H*K + o_k
    p_kj = k + i_ti * H*K + o_k
    p_gkj = g + i_ti * HV*K + o_k
    p_bj = beta + i_ti * HV
    # use the same anchor for key-side gradients so the shared offset cancels
    if matrix_ok:
        b_gn = center[None, :]
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

        b_dkt += tl.dot(b_dAqk_diag_kk, b_q_exp, input_precision='tf32x3') * b_gk_diag_kk
        b_dkt += tl.dot(b_dAkk_diag_kk, b_kb_exp, input_precision='tf32x3') * b_gk_diag_kk
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
            b_gkq = exp2(tl.where(m_ck & m_i, b_gkj[None, :] - b_g, 0.))
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


class EndpointMatrixIntra:
    def __init__(self,original,residual_original,beta_complete=False,warps=4):
        self.original=residual_original if beta_complete else original
        self.beta_complete,self.warps=beta_complete,warps
        self.flags=None;self.kernel=None;self.calls=0;self.fallback_calls=0

    def __call__(self,q,k,g,beta,dAqk,dAkk,dq,dk,db,dg,cu_seqlens=None,
                 chunk_indices=None,chunk_size=64,safe_gate=False,lower_bound=None,use_graph=False):
        self.calls+=1
        if (use_graph or chunk_size not in (32,64) or safe_gate or
            not all(x.is_contiguous() for x in (q,k,g,beta,dAqk,dAkk,dq,dk,db,dg))):
            self.fallback_calls+=1;self.flags=None
            return self.original(q=q,k=k,g=g,beta=beta,dAqk=dAqk,dAkk=dAkk,
                                 dq=dq,dk=dk,db=db,dg=dg,cu_seqlens=cu_seqlens,
                                 chunk_indices=chunk_indices,chunk_size=chunk_size,
                                 safe_gate=safe_gate,lower_bound=lower_bound,use_graph=use_graph)
        B,T,H,K=k.shape;HV=g.shape[2]
        if chunk_indices is None and cu_seqlens is not None:
            chunk_indices=prepare_chunk_indices(cu_seqlens,chunk_size)
        nt=triton.cdiv(T,chunk_size) if chunk_indices is None else len(chunk_indices)
        grid=(nt*(chunk_size//16)*triton.cdiv(K,32),B*HV)
        self.flags=torch.empty(grid,device=q.device,dtype=torch.int32)
        self.kernel=endpoint_matrix_kernel[grid](q,k,g,beta,dAqk,dAkk,dq,dq,dk,dk,dg,dg,db,self.flags,
            cu_seqlens,chunk_indices,B,T,H=H,HV=HV,K=K,BT=chunk_size,BK=32,BC=16,NC=chunk_size//16,
            IS_VARLEN=cu_seqlens is not None,SAFE_GATE=True,BETA_COMPLETE=self.beta_complete,
            GATE_SHIFT=0.,USE_GATHER=False,USE_GRAPH=False,num_warps=self.warps,num_stages=2)
        return dq,dk,db,dg
