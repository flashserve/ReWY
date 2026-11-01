# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Derived from frozen FLA under its MIT license; arithmetic and dtypes unchanged.
"""950 GDN2 finalize compatibility: one task per program for all shapes."""
import triton
import triton.language as tl
from .kda_wy_finalize_npu950 import Finalize950

@triton.jit(do_not_specialize=['T', 'task_num', 'num_core', 'BH', 'NT_OFFSET'])
def gdn2_wy_finalize_950(A, dA_acc, dA, cu_seqlens, chunk_indices, T, BH, task_num, num_core, NT_OFFSET, H: tl.constexpr, BT: tl.constexpr, IS_VARLEN: tl.constexpr, A_T_CONTIG: tl.constexpr, TAIL_MODE: tl.constexpr, TASK_OFFSET: tl.constexpr=0):
    core_id = tl.program_id(0)
    T_seq = T
    task_id = tl.program_id(0) + TASK_OFFSET
    i_t = NT_OFFSET + task_id // BH
    i_bh = task_id % BH
    i_b, i_h = (i_bh // H, i_bh % H)
    if IS_VARLEN:
        i_n, i_t = (tl.load(chunk_indices + i_t * 2).to(tl.int32), tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32))
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64))
        T = (eos - bos).to(tl.int32)
    else:
        bos = tl.cast(i_b, tl.int64) * T
    if A_T_CONTIG:
        if IS_VARLEN:
            A_ptr = A + (tl.cast(i_h, tl.int64) * T_seq + bos) * BT
        else:
            A_ptr = A + (tl.cast(i_b, tl.int64) * H + i_h) * T_seq * BT
        a_stride_t = BT
    else:
        A_ptr = A + (bos * H + i_h) * BT
        a_stride_t = H * BT
    dA_acc_ptr = dA_acc + (bos * H + i_h) * BT
    dA_ptr = dA + (bos * H + i_h) * BT
    p_A = tl.make_block_ptr(A_ptr, (BT, T), (1, a_stride_t), (0, i_t * BT), (BT, BT), (0, 1))
    p_dA_acc = tl.make_block_ptr(dA_acc_ptr, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    p_dA = tl.make_block_ptr(dA_ptr, (T, BT), (H * BT, 1), (i_t * BT, 0), (BT, BT), (1, 0))
    o_t = i_t * BT + tl.arange(0, BT)
    if TAIL_MODE == 0:
        b_A = tl.load(p_A)
        b_dA = tl.load(p_dA_acc).to(tl.float32)
        m_A = o_t[:, None] > o_t[None, :]
    else:
        b_A = tl.load(p_A, boundary_check=(0, 1))
        b_dA = tl.load(p_dA_acc, boundary_check=(0, 1)).to(tl.float32)
        m_t = o_t < T
        m_A = (o_t[:, None] > o_t[None, :]) & (m_t[:, None] & m_t[None, :])
    b_dA = tl.where(m_A, b_dA, 0)
    b_mid = tl.dot(b_dA.to(b_A.dtype), b_A, allow_tf32=False)
    b_fin = tl.dot(b_A, b_mid.to(b_A.dtype), allow_tf32=False)
    b_fin = tl.where(m_A, -b_fin, 0)
    if TAIL_MODE == 0:
        tl.store(p_dA, b_fin.to(p_dA.dtype.element_ty))
    else:
        tl.store(p_dA, b_fin.to(p_dA.dtype.element_ty), boundary_check=(0, 1))

def install():
    import importlib
    from fast_delta_rule.backends.npu_target import require_950
    require_950('npu:0')
    module=importlib.import_module('fla.ops.gdn2.backends.triton_ascend.chunk_bwd')
    module.chunk_gdn2_bwd_kernel_wy_dA_finalize_npu=Finalize950(gdn2_wy_finalize_950)
    return dict(id='gdn2-wy-finalize-one-task-per-program',
                reason='Public backward stage capture found nondeterministic dAkk at the WY finalize boundary on 950',
                scope='all shapes and both original/matrix-only WY',
                precision='same masks, operand casts and FP32 accumulators/output',
                frozen_disk_source_modified=False)
