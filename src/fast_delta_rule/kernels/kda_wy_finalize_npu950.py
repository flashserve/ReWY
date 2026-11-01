# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Derived from frozen FLA under its MIT license; arithmetic and dtypes retained.
"""950 compatibility: one WY finalize/dw task per program, avoiding persistent-loop corruption.

Same two products, masks, operand casts, accumulator and output precision as FLA.
The host splits only for the hardware grid limit, uniformly for every shape.
"""
import triton
import triton.language as tl
from fla.ops.utils.op import exp2
from fla.utils.ascend_ub_manager import ASCEND_MAX_GRID_DIM

@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH', 'NT_OFFSET'])
def kda_wy_finalize_950(A, beta, dA_acc, db_acc, dA, db, cu_seqlens, chunk_indices, T, BH, task_num, num_core, NT_OFFSET, HV: tl.constexpr, BT: tl.constexpr, IS_VARLEN: tl.constexpr, G_T_CONTIG: tl.constexpr, TAIL_MODE: tl.constexpr, USE_GRAPH: tl.constexpr=False, TASK_OFFSET: tl.constexpr=0):
    """dA = mask(-A @ ((mask * dA_acc * beta) @ A)); copy db_acc into db.

    Fuses the old mask kernel into mid+finalize so dA_acc stays in UB.
    TAIL_MODE 0 = aligned bulk (no boundary_check). TAIL_MODE 1 = tail/varlen.
    First tl.dot clobbers masked dA (dead). Second uses b_A as lhs (dead after store).
    """
    core_id = tl.program_id(0)
    T_seq = T
    task_id = tl.program_id(0) + TASK_OFFSET
    i_t = NT_OFFSET + task_id // BH
    i_bh = task_id % BH
    i_b, i_hv = (i_bh // HV, i_bh % HV)
    is_valid = True
    if IS_VARLEN:
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        if USE_GRAPH:
            is_valid = i_n >= 0
    if is_valid:
        if IS_VARLEN:
            bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64))
            T = (eos - bos).to(tl.int32)
        else:
            bos, eos = (tl.cast(i_b, tl.int64) * T, tl.cast(i_b, tl.int64) * T + T)
        if G_T_CONTIG:
            if IS_VARLEN:
                A_ptr = A + tl.cast(i_hv, tl.int64) * T_seq * BT + bos * BT
                beta_ptr = beta + tl.cast(i_hv, tl.int64) * T_seq + bos
            else:
                hv_off = tl.cast(i_b, tl.int64) * HV + i_hv
                A_ptr = A + hv_off * T_seq * BT
                beta_ptr = beta + hv_off * T_seq
            a_stride_t = BT
            beta_stride = 1
        else:
            A_ptr = A + (bos * HV + i_hv) * BT
            beta_ptr = beta + bos * HV + i_hv
            a_stride_t = HV * BT
            beta_stride = HV
        dA_acc_ptr = dA_acc + (bos * HV + i_hv) * BT
        db_acc_ptr = db_acc + bos * HV + i_hv
        dA_ptr = dA + (bos * HV + i_hv) * BT
        db_ptr = db + bos * HV + i_hv
        p_A = tl.make_block_ptr(A_ptr, (BT, T), (1, a_stride_t), (0, i_t * BT), (BT, BT), (0, 1))
        p_dA_acc = tl.make_block_ptr(dA_acc_ptr, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_dA = tl.make_block_ptr(dA_ptr, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        p_beta = tl.make_block_ptr(beta_ptr, (T,), (beta_stride,), (i_t * BT,), (BT,), (0,))
        p_db_acc = tl.make_block_ptr(db_acc_ptr, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        p_db = tl.make_block_ptr(db_ptr, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        o_t = i_t * BT + tl.arange(0, BT)
        if TAIL_MODE == 0:
            b_A = tl.load(p_A)
            b_dA = tl.load(p_dA_acc).to(tl.float32)
            b_beta = tl.load(p_beta)
            m_A = o_t[:, None] > o_t[None, :]
        else:
            b_A = tl.load(p_A, boundary_check=(0, 1))
            b_dA = tl.load(p_dA_acc, boundary_check=(0, 1)).to(tl.float32)
            b_beta = tl.load(p_beta, boundary_check=(0,))
            m_t = o_t < T
            m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t[None, :])
        b_dA = tl.where(m_A, b_dA * b_beta[None, :], 0)
        b_mid = tl.dot(b_dA.to(b_A.dtype), b_A, allow_tf32=False)
        b_fin = tl.dot(b_A, b_mid.to(b_A.dtype), allow_tf32=False)
        b_fin = tl.where(m_A, -b_fin, 0)
        if TAIL_MODE == 0:
            tl.store(p_dA, b_fin.to(p_dA.dtype.element_ty))
            tl.store(p_db, tl.load(p_db_acc).to(p_db.dtype.element_ty))
        else:
            tl.store(p_dA, b_fin.to(p_dA.dtype.element_ty), boundary_check=(0, 1))
            tl.store(p_db, tl.load(p_db_acc, boundary_check=(0,)).to(p_db.dtype.element_ty), boundary_check=(0,))

@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH'])
def kda_wy_dw_950(k, g, beta, A, h, dv, dA_acc, db_acc, dg, dk, cu_seqlens, chunk_indices, chunk_offsets, T, BH, task_num, num_core, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, STATE_V_FIRST: tl.constexpr, IS_VARLEN: tl.constexpr, K_T_CONTIG: tl.constexpr, G_T_CONTIG: tl.constexpr, K_OFFSET: tl.constexpr, USE_GRAPH: tl.constexpr=False, TASK_OFFSET: tl.constexpr=0):
    i_k = K_OFFSET
    core_id = tl.program_id(0)
    T_seq = T
    task_id = tl.program_id(0) + TASK_OFFSET
    i_t = task_id // BH
    i_bh = task_id % BH
    i_b, i_hv = (i_bh // HV, i_bh % HV)
    i_h = i_hv // (HV // H)
    is_valid = True
    if IS_VARLEN:
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        if USE_GRAPH:
            is_valid = i_n >= 0
    if is_valid:
        if IS_VARLEN:
            bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64))
            T = (eos - bos).to(tl.int32)
            i_tg = tl.load(chunk_offsets + i_n).to(tl.int64) + i_t.to(tl.int64)
        else:
            i_tg = tl.cast(i_b, tl.int64) * tl.cdiv(T, BT) + i_t
            bos, eos = (tl.cast(i_b, tl.int64) * T, tl.cast(i_b, tl.int64) * T + T)
        if K_T_CONTIG:
            if IS_VARLEN:
                k_ptr = k + tl.cast(i_h, tl.int64) * T_seq * K + bos * K
            else:
                k_ptr = k + (tl.cast(i_b, tl.int64) * H + i_h) * T_seq * K
            k_stride_t = K
        else:
            k_ptr = k + (bos * H + i_h) * K
            k_stride_t = H * K
        if G_T_CONTIG:
            if IS_VARLEN:
                g_ptr = g + tl.cast(i_hv, tl.int64) * T_seq * K + bos * K
                beta_ptr = beta + tl.cast(i_hv, tl.int64) * T_seq + bos
                A_ptr = A + tl.cast(i_hv, tl.int64) * T_seq * BT + bos * BT
                dv_ptr = dv + tl.cast(i_hv, tl.int64) * T_seq * V + bos * V
            else:
                hv_off = tl.cast(i_b, tl.int64) * HV + i_hv
                g_ptr = g + hv_off * T_seq * K
                beta_ptr = beta + hv_off * T_seq
                A_ptr = A + hv_off * T_seq * BT
                dv_ptr = dv + hv_off * T_seq * V
            g_stride_t = K
            a_stride_t = BT
            dv_stride_t = V
            beta_stride = 1
        else:
            g_ptr = g + (bos * HV + i_hv) * K
            beta_ptr = beta + bos * HV + i_hv
            A_ptr = A + (bos * HV + i_hv) * BT
            dv_ptr = dv + (bos * HV + i_hv) * V
            g_stride_t = HV * K
            a_stride_t = HV * BT
            dv_stride_t = HV * V
            beta_stride = HV
        h_ptr = h + (i_tg * HV + i_hv) * K * V
        dA_ptr = dA_acc + (bos * HV + i_hv) * BT
        db_ptr = db_acc + bos * HV + i_hv
        dg_ptr = dg + (bos * HV + i_hv) * K
        dk_ptr = dk + (bos * HV + i_hv) * K
        b_dw = tl.zeros([BT, BK], dtype=tl.float32)
        for i_v in range(tl.cdiv(V, BV)):
            p_dv = tl.make_block_ptr(dv_ptr, (T, V), (dv_stride_t, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
            if STATE_V_FIRST:
                p_h = tl.make_block_ptr(h_ptr, (V, K), (K, 1), (i_v * BV, i_k * BK), (BV, BK), (1, 0))
            else:
                p_h = tl.make_block_ptr(h_ptr, (V, K), (1, V), (i_v * BV, i_k * BK), (BV, BK), (0, 1))
            b_dv = tl.load(p_dv, boundary_check=(0, 1))
            b_h = tl.load(p_h, boundary_check=(0, 1))
            b_dw = tl.dot(b_dv, b_h.to(b_dv.dtype), b_dw, allow_tf32=False)
        p_k = tl.make_block_ptr(k_ptr, (T, K), (k_stride_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_g = tl.make_block_ptr(g_ptr, (T, K), (g_stride_t, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        p_beta = tl.make_block_ptr(beta_ptr, (T,), (beta_stride,), (i_t * BT,), (BT,), (0,))
        p_A = tl.make_block_ptr(A_ptr, (BT, T), (1, a_stride_t), (0, i_t * BT), (BT, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))
        b_g = tl.load(p_g, boundary_check=(0, 1)).to(tl.float32)
        b_beta = tl.load(p_beta, boundary_check=(0,))
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_gk_exp = exp2(b_g)
        b_kg = b_k * b_gk_exp
        b_gb = b_gk_exp * b_beta[:, None]
        b_dw = -b_dw.to(b_A.dtype)
        b_kg_a = b_kg.to(b_A.dtype)
        b_dkgb = tl.dot(b_A, b_dw, allow_tf32=False)
        p_dA_acc = tl.make_block_ptr(dA_ptr, (T, BT), (HV * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
        b_dA = tl.load(p_dA_acc, boundary_check=(0, 1)).to(tl.float32)
        b_dw_c = b_dw + 0.0
        b_dA = tl.dot(b_dw_c, tl.trans(b_kg_a), b_dA, allow_tf32=False)
        tl.store(p_dA_acc, b_dA.to(p_dA_acc.dtype.element_ty), boundary_check=(0, 1))
        p_db_acc = tl.make_block_ptr(db_ptr, (T,), (HV,), (i_t * BT,), (BT,), (0,))
        b_db = tl.load(p_db_acc, boundary_check=(0,)).to(tl.float32)
        b_db += tl.sum(b_dkgb * b_kg, 1)
        tl.store(p_db_acc, b_db.to(p_db_acc.dtype.element_ty), boundary_check=(0,))
        p_dk = tl.make_block_ptr(dk_ptr, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_dk = tl.load(p_dk, boundary_check=(0, 1)).to(tl.float32)
        b_dk = b_dk + b_dkgb * b_gb
        tl.store(p_dk, b_dk.to(p_dk.dtype.element_ty), boundary_check=(0, 1))
        p_dg = tl.make_block_ptr(dg_ptr, (T, K), (HV * K, 1), (i_t * BT, i_k * BK), (BT, BK), (1, 0))
        b_dg = tl.load(p_dg, boundary_check=(0, 1)).to(tl.float32)
        b_dg = b_dg + b_kg * b_dkgb * b_beta[:, None]
        tl.store(p_dg, b_dg.to(p_dg.dtype.element_ty), boundary_check=(0, 1))

class Finalize950:
    def __init__(self, kernel=kda_wy_finalize_950):
        self.kernel=kernel

    def __getitem__(self, unused_persistent_grid):
        def launch(**kwargs):
            tasks = kwargs['task_num']
            for offset in range(0, tasks, ASCEND_MAX_GRID_DIM):
                count = min(tasks-offset, ASCEND_MAX_GRID_DIM)
                self.kernel[(count,)](TASK_OFFSET=offset, **kwargs)
        return launch

def install():
    from fast_delta_rule.backends.npu_target import require_950
    require_950('npu:0')
    import importlib
    module = importlib.import_module('fla.ops.kda.backends.triton_ascend.chunk_bwd')
    module.chunk_kda_bwd_kernel_wy_dA_finalize_npu = Finalize950()
    module.chunk_kda_bwd_kernel_wy_dw_part_npu = Finalize950(kda_wy_dw_950)
    return dict(id='kda-wy-finalize-and-dw-one-task-per-program',
                reason='Persistent finalize and dw loops corrupt dAkk/dg on Ascend950DT; fixed-input stage replay proved nondeterminism',
                scope='all KDA shapes; original and optimized matrix-only WY share this compatibility schedule',
                precision='frozen equations, operand casts, FP32 accumulators and outputs unchanged',
                frozen_disk_source_modified=False, grid_limit=ASCEND_MAX_GRID_DIM)
