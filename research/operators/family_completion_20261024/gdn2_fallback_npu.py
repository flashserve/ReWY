# Adapted from frozen FLA, MIT license. Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li.
# Formula/subchunk implementation unchanged. Only flattened-id helper and GPU
# autotune launch are replaced with a persistent Ascend core loop.
from common import torch,triton
import triton.language as tl
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.op import exp2,gather
from fla.utils import IS_GATHER_SUPPORTED,ascend_compile_kwargs
from fla.utils.ascend_ub_manager import get_npu_properties
SAFE_GATE_DOT_PRECISION=tl.constexpr('ieee')

# Candidate fallback: standard-layout additive db and in-place other outputs.
@triton.jit
def body(I_KC,I_T,I_BH,
    q,
    k,
    g,
    b,
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
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    NC: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    SAFE_GATE: tl.constexpr,
    GATE_SHIFT: tl.constexpr,
    USE_GATHER: tl.constexpr,
):
    i_kc, i_t = I_KC, I_T
    i_bh = I_BH.to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    i_k, i_i = i_kc // NC, i_kc % NC

    all = tl.cast(B,tl.int64) * T
    if IS_VARLEN:
        i_n, i_t = tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64)
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    else:
        bos, eos = i_b * T, i_b * T + T
    T = eos - bos

    i_ti = i_t * BT + i_i * BC
    if i_ti < T:

        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        q += (bos * H + i_h) * K
        k += (bos * H + i_h) * K
        g += (bos * H + i_h) * K
        b += (bos * H + i_h) * K

        dAqk += (bos * H + i_h) * BT
        dAkk += (bos * H + i_h) * BT
        dq += (bos * H + i_h) * K
        dq2 += (bos * H + i_h) * K
        dk += (bos * H + i_h) * K
        dk2 += (bos * H + i_h) * K
        dg += (bos * H + i_h) * K
        dg2 += (bos * H + i_h) * K
        db += (bos * H + i_h) * K

        o_i = tl.arange(0, BC)
        o_c = i_ti + o_i
        m_c = o_c < T
        m_kc = m_c[:, None] & m_k[None, :]
        p_g = g + o_c[:, None] * (H * K) + o_k[None, :]
        b_g = tl.load(p_g, mask=m_kc, other=0.0).to(tl.float32)

        p_b = b + o_c[:, None] * (H * K) + o_k[None, :]
        b_b = tl.load(p_b, mask=m_kc, other=0.0)

        b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
        b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)

        if i_i > 0:
            p_gn = g + i_ti * H * K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
            for i_j in range(0, i_i):
                o_j = i_t * BT + i_j * BC + o_i
                m_j = o_j < T
                m_jk = m_j[:, None] & m_k[None, :]
                m_dj = m_c[:, None] & ((i_j * BC + o_i)[None, :] < BT)
                p_k = k + o_j[:, None] * (H * K) + o_k[None, :]
                p_gk = g + o_j[:, None] * (H * K) + o_k[None, :]
                p_dAqk = dAqk + o_c[:, None] * (H * BT) + (i_j * BC + o_i)[None, :]
                p_dAkk = dAkk + o_c[:, None] * (H * BT) + (i_j * BC + o_i)[None, :]
                b_kj = tl.load(p_k, mask=m_jk, other=0.0)
                b_gk = tl.load(p_gk, mask=m_jk, other=0.0)
                b_kg = b_kj * exp2(b_gn - b_gk)
                b_dAqk = tl.load(p_dAqk, mask=m_dj, other=0.0)
                b_dAkk = tl.load(p_dAkk, mask=m_dj, other=0.0)
                b_dq2 = tl.dot(b_dAqk, b_kg, b_dq2)
                b_dk2 = tl.dot(b_dAkk, b_kg, b_dk2)
            b_gqn = exp2(b_g - b_gn)
            b_dq2 *= b_gqn
            b_dk2 *= b_gqn

        m_dA = (i_ti + o_i) < T
        o_dA = (i_ti + o_i) * H * BT + i_i * BC
        p_kj = k + i_ti * H * K + o_k
        p_gkj = g + i_ti * H * K + o_k

        p_q = q + o_c[:, None] * (H * K) + o_k[None, :]
        p_k = k + o_c[:, None] * (H * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_kc, other=0.0)
        b_k = tl.load(p_k, mask=m_kc, other=0.0)

        if SAFE_GATE:
            if USE_GATHER:
                b_gn = gather(b_g, tl.zeros([1, BK], dtype=tl.int16), axis=0)
            else:
                p_gn = g + i_ti * H * K + o_k
                b_gn = tl.load(p_gn, mask=m_k, other=0)[None, :]
            b_gn = b_gn - 0.5 * (BC - 1) * GATE_SHIFT

            m_dd = m_c[:, None] & ((i_i * BC + o_i)[None, :] < BT)
            p_dAqk = dAqk + o_c[:, None] * (H * BT) + (i_i * BC + o_i)[None, :]
            p_dAkk = dAkk + o_c[:, None] * (H * BT) + (i_i * BC + o_i)[None, :]
            b_dAqk_diag_qk = tl.load(p_dAqk, mask=m_dd, other=0.0).to(tl.float32)
            b_dAkk_diag_qk = tl.load(p_dAkk, mask=m_dd, other=0.0).to(tl.float32)

            m_i_diag_qk = (o_i[:, None] >= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
            m_j_diag_qk = (i_ti + o_i[:, None]) < T

            b_dAqk_diag_qk = tl.where(m_i_diag_qk, b_dAqk_diag_qk, 0.)
            b_dAkk_diag_qk = tl.where(m_i_diag_qk, b_dAkk_diag_qk, 0.)
            b_g_diag_qk = tl.where(m_j_diag_qk, b_g - b_gn, 0.)
            exp_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(b_g_diag_qk), 0.)
            exp_neg_b_g_diag_qk = tl.where(m_j_diag_qk, exp2(-b_g_diag_qk), 0.)

            b_k_exp_diag_qk = b_k * exp_neg_b_g_diag_qk
            b_dq2 += tl.dot(b_dAqk_diag_qk, b_k_exp_diag_qk, input_precision=SAFE_GATE_DOT_PRECISION) * exp_b_g_diag_qk
            b_dk2 += tl.dot(b_dAkk_diag_qk, b_k_exp_diag_qk, input_precision=SAFE_GATE_DOT_PRECISION) * exp_b_g_diag_qk
        else:
            for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
                b_dAqk = tl.load(dAqk + o_dA + j, mask=m_dA, other=0)
                b_dAkk = tl.load(dAkk + o_dA + j, mask=m_dA, other=0)
                b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
                b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
                m_i = o_i[:, None] >= j
                b_gqk = exp2(b_g - b_gkj[None, :])
                b_dq2 += tl.where(m_i, b_dAqk[:, None] * b_kj[None, :] * b_gqk, 0.)
                b_dk2 += tl.where(m_i, b_dAkk[:, None] * b_kj[None, :] * b_gqk, 0.)

                p_kj += H * K
                p_gkj += H * K

        # db gets the elementwise b * k contribution from dk2 (before b is folded
        # back into dk2 itself); then dk2 is rescaled by the channel-wise gate.
        b_db_tile = b_dk2 * b_k
        b_dk2 = b_dk2 * b_b

        o_db = tl.arange(0, BK)
        m_db = m_kc
        p_dq = dq + o_c[:, None] * (H * K) + o_k[None, :]
        p_dq2 = dq2 + o_c[:, None] * (H * K) + o_k[None, :]
        p_db = db + o_c[:, None] * (H * K) + o_k[None, :]

        b_dg2 = b_q * b_dq2
        b_dq2 = b_dq2 + tl.load(p_dq, mask=m_kc, other=0.0)
        tl.store(p_dq2, b_dq2.to(p_dq2.dtype.element_ty), mask=m_kc)
        tl.store(p_db, tl.load(p_db, mask=m_db, other=0.0) + b_db_tile, mask=m_db)

        tl.debug_barrier()
        b_dkt = tl.zeros([BC, BK], dtype=tl.float32)

        valid_nc = min(NC, tl.cdiv(T - i_t * BT, BC))
        if i_i < valid_nc - 1:
            p_gn = g + (min(i_ti + BC, T) - 1) * H * K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
            for i_j in range(i_i + 1, valid_nc):
                o_j = i_t * BT + i_j * BC + o_i
                m_j = o_j < T
                m_jk = m_j[:, None] & m_k[None, :]
                m_dj = ((i_i * BC + o_i)[:, None] < BT) & m_j[None, :]
                p_q = q + o_j[:, None] * (H * K) + o_k[None, :]
                p_k = k + o_j[:, None] * (H * K) + o_k[None, :]
                p_gk = g + o_j[:, None] * (H * K) + o_k[None, :]
                p_bj = b + o_j[:, None] * (H * K) + o_k[None, :]
                p_dAqk = dAqk + (i_i * BC + o_i)[:, None] + o_j[None, :] * (H * BT)
                p_dAkk = dAkk + (i_i * BC + o_i)[:, None] + o_j[None, :] * (H * BT)
                b_bj = tl.load(p_bj, mask=m_jk, other=0.0)
                b_qj = tl.load(p_q, mask=m_jk, other=0.0)
                b_kbj = tl.load(p_k, mask=m_jk, other=0.0) * b_bj
                b_gk = tl.load(p_gk, mask=m_jk, other=0.0).to(tl.float32)
                b_dAqk = tl.load(p_dAqk, mask=m_dj, other=0.0)
                b_dAkk = tl.load(p_dAkk, mask=m_dj, other=0.0)
                b_gkn = exp2(b_gk - b_gn)
                b_qg = b_qj * tl.where(m_j[:, None], b_gkn, 0)
                b_kbg = b_kbj * tl.where(m_j[:, None], b_gkn, 0)
                b_dkt = tl.dot(b_dAqk, b_qg, b_dkt)
                b_dkt = tl.dot(b_dAkk, b_kbg, b_dkt)
            b_dkt *= exp2(b_gn - b_g)

        o_dA = i_ti * H * BT + i_i * BC + o_i
        p_qj = q + i_ti * H * K + o_k
        p_kj = k + i_ti * H * K + o_k
        p_gkj = g + i_ti * H * K + o_k
        p_bj_ptr = b + i_ti * H * K + o_k

        if SAFE_GATE:
            if USE_GATHER:
                b_gn = gather(b_g, tl.zeros([1, BK], dtype=tl.int16), axis=0)
            else:
                p_gn = g + i_ti * H * K + o_k
                b_gn = tl.load(p_gn, mask=m_k, other=0).to(tl.float32)[None, :]
            b_gn = b_gn - 0.5 * (BC - 1) * GATE_SHIFT
            p_q = q + o_c[:, None] * (H * K) + o_k[None, :]
            b_q = tl.load(p_q, mask=m_kc, other=0.0)
            m_dk = ((i_i * BC + o_i)[:, None] < BT) & m_c[None, :]
            p_dAqk = dAqk + (i_i * BC + o_i)[:, None] + o_c[None, :] * (H * BT)
            p_dAkk = dAkk + (i_i * BC + o_i)[:, None] + o_c[None, :] * (H * BT)
            b_dAqk_diag_kk = tl.load(p_dAqk, mask=m_dk, other=0.0).to(tl.float32)
            b_dAkk_diag_kk = tl.load(p_dAkk, mask=m_dk, other=0.0).to(tl.float32)

            m_i_diag_kk = (o_i[:, None] <= o_i[None, :]) & ((i_ti + o_i[:, None]) < T) & ((i_ti + o_i[None, :]) < T)
            m_j_diag_kk = (i_ti + o_i[:, None]) < T

            b_dAqk_diag_kk = tl.where(m_i_diag_kk, b_dAqk_diag_kk, 0.)
            b_dAkk_diag_kk = tl.where(m_i_diag_kk, b_dAkk_diag_kk, 0.)
            b_g_diag_kk = tl.where(m_j_diag_kk, b_g - b_gn, 0.)
            exp_b_g_diag_kk = tl.where(m_j_diag_kk, exp2(b_g_diag_kk), 0.)
            exp_neg_b_g_diag_kk = tl.where(m_j_diag_kk, exp2(-b_g_diag_kk), 0.)

            b_q_exp = b_q * exp_b_g_diag_kk
            b_kb_exp = b_k * b_b * exp_b_g_diag_kk

            b_dkt += tl.dot(b_dAqk_diag_kk, b_q_exp, input_precision=SAFE_GATE_DOT_PRECISION) * exp_neg_b_g_diag_kk
            b_dkt += tl.dot(b_dAkk_diag_kk, b_kb_exp, input_precision=SAFE_GATE_DOT_PRECISION) * exp_neg_b_g_diag_kk
        else:
            for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
                b_dAqk = tl.load(dAqk + o_dA + j * H * BT)
                b_dAkk = tl.load(dAkk + o_dA + j * H * BT)
                b_qj = tl.load(p_qj, mask=m_k, other=0).to(tl.float32)
                b_kj = tl.load(p_kj, mask=m_k, other=0).to(tl.float32)
                b_bj = tl.load(p_bj_ptr, mask=m_k, other=0).to(tl.float32)
                b_kbj = b_kj * b_bj
                b_gkj = tl.load(p_gkj, mask=m_k, other=0).to(tl.float32)
                m_i = o_i[:, None] <= j
                b_gkq = exp2(b_gkj[None, :] - b_g)
                b_dkt += tl.where(m_i, b_dAqk[:, None] * b_qj[None, :] * b_gkq, 0.)
                b_dkt += tl.where(m_i, b_dAkk[:, None] * b_kbj[None, :] * b_gkq, 0.)

                p_qj += H * K
                p_kj += H * K
                p_gkj += H * K
                p_bj_ptr += H * K

        p_dk = dk + o_c[:, None] * (H * K) + o_k[None, :]
        p_dk2 = dk2 + o_c[:, None] * (H * K) + o_k[None, :]
        p_dg = dg + o_c[:, None] * (H * K) + o_k[None, :]
        p_dg2 = dg2 + o_c[:, None] * (H * K) + o_k[None, :]

        b_dg2 += (b_dk2 - b_dkt) * b_k + tl.load(p_dg, mask=m_kc, other=0.0)
        b_dk2 += tl.load(p_dk, mask=m_kc, other=0.0)
        b_dk2 += b_dkt

        tl.store(p_dk2, b_dk2.to(p_dk2.dtype.element_ty), mask=m_kc)
        tl.store(p_dg2, b_dg2.to(p_dg2.dtype.element_ty), mask=m_kc)
