# Adapted from frozen FLA (MIT), Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li.
# Preserve Ascend data movement and task ownership; use bounded FP32 matrix diagonal.
import torch,triton
import triton.language as tl
from fla.ops.kda.backends.triton_ascend.chunk_intra import get_npu_properties
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.graph import get_static_buffer
from fla.ops.utils.op import exp2
from fla.utils import input_guard,ascend_compile_kwargs

@triton.jit(do_not_specialize=['B', 'T', 'NT_TOTAL'])
def chunk_kda_bwd_kernel_intra_npu(
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
    SAFE_GATE: tl.constexpr,
    NT_TOTAL,
    USE_GRAPH: tl.constexpr = False,
):
    NC = tl.cdiv(BT, BC)
    i_core = tl.program_id(0)
    num_core = tl.num_programs(0)
    # widen before multiplying; the task decomposition stays in int64 end to end
    NK = tl.cdiv(K, BK)
    task_num = tl.cast(NT_TOTAL, tl.int64) * NC * B * HV * NK
    BH_TOTAL = B * HV
    for i_task in tl.range(i_core, task_num, num_core):
        i_kslice = i_task % NK
        i_logical = i_task // NK
        i_bh = i_logical % BH_TOTAL
        i_tc = i_logical // BH_TOTAL
        i_t = i_tc % NT_TOTAL
        i_i0 = i_tc // NT_TOTAL
        i_b, i_hv = i_bh // HV, i_bh % HV
        i_h = i_hv // (HV // H)

        m_valid = True
        if cu_seqlens is not None:
            i_n, i_t = tl.load(chunk_indices + i_t * 2), tl.load(chunk_indices + i_t * 2 + 1)
            if USE_GRAPH:
                # graph padding uses [-1, 0]; exclude it before sequence-dependent memory accesses.
                m_valid = i_n >= 0
        # skip only this task: returning would drop later valid tasks assigned to this core.
        if m_valid:
            if cu_seqlens is not None:
                # int64 guarantees: cu_seqlens may arrive as int32 and chunk_indices
                # follows its dtype, but the global offsets must stay 64-bit
                bos, eos = tl.cast(tl.load(cu_seqlens + i_n), tl.int64), tl.cast(tl.load(cu_seqlens + i_n + 1), tl.int64)
            else:
                bos, eos = i_b * T, i_b * T + T
            # T is a loop-carried arg (int32); the reassignment must keep its type.
            T = tl.cast(eos - bos, tl.int32)

            # rebind pointers per task (ptr-arg += inside the task loop would both
            # accumulate offsets across tasks and break loop-carried typecheck)
            o_h = (bos * H + i_h) * K
            o_hv = bos * HV + i_hv
            p_q = q + o_h
            p_k = k + o_h
            p_g = g + o_hv * K
            p_beta = beta + o_hv
            p_dAqk = dAqk + o_hv * BT
            p_dAkk = dAkk + o_hv * BT
            p_dq = dq + o_hv * K
            p_dq2 = dq2 + o_hv * K
            p_dk = dk + o_hv * K
            p_dk2 = dk2 + o_hv * K
            p_dg = dg + o_hv * K
            p_dg2 = dg2 + o_hv * K
            p_db = db + i_kslice * B * T * HV + o_hv

            o_k = i_kslice * BK + tl.arange(0, BK)
            o_i = tl.arange(0, BC)
            m_k = o_k < K
            NC_LOC = min(NC, tl.cdiv(T - i_t * BT, BC))
            i_i = i_i0
            if i_i0 < NC_LOC:
                # no-op cast when i_t is int64; guards int32 chunk_indices tables
                i_ti = tl.cast(i_t * BT + i_i * BC, tl.int64)
                m_row = (i_ti + o_i) < T
                m_ik = m_row[:, None] & m_k[None, :]
                o_row = i_i * BC + o_i

                b_g = tl.load(p_g + i_ti * (HV * K) + o_i[:, None] * (HV * K) +
                              o_k[None, :], mask=m_ik, other=0.0).to(tl.float32)
                b_b = tl.load(p_beta + i_ti * HV + o_i * HV, mask=m_row, other=0.0)
                b_q = tl.load(p_q + i_ti * (H * K) + o_i[:, None] * (H * K) + o_k[None, :], mask=m_ik, other=0.0)
                b_k = tl.load(p_k + i_ti * (H * K) + o_i[:, None] * (H * K) + o_k[None, :], mask=m_ik, other=0.0)

                b_dq2 = tl.zeros([BC, BK], dtype=tl.float32)
                b_dk2 = tl.zeros([BC, BK], dtype=tl.float32)

                # ---- inter blocks (j < i) ----
                if i_i > 0:
                    b_gn = tl.load(p_g + i_ti * HV * K + o_k, mask=m_k, other=0.0).to(tl.float32)[None, :]
                    for i_j in range(0, i_i):
                        i_tj = tl.cast(i_t * BT + i_j * BC, tl.int64)
                        m_rowj = (i_tj + o_i) < T
                        m_ikj = m_rowj[:, None] & m_k[None, :]
                        b_kj = tl.load(p_k + i_tj * (H * K) + o_i[:, None] * (H * K) + o_k[None, :], mask=m_ikj, other=0.0)
                        b_gkj = tl.load(p_g + i_tj * (HV * K) + o_i[:, None] * (HV * K) + o_k[None, :], mask=m_ikj, other=0.0)
                        b_kg = b_kj * exp2(b_gn - b_gkj.to(tl.float32))
                        m_ij = m_row[:, None] & ((i_j * BC + o_i)[None, :] < BT)
                        b_dAqk = tl.load(p_dAqk + i_ti * (HV * BT) + o_i[:, None] * (HV * BT) +
                                         (i_j * BC + o_i)[None, :], mask=m_ij, other=0.0)
                        b_dAkk = tl.load(p_dAkk + i_ti * (HV * BT) + o_i[:, None] * (HV * BT) +
                                         (i_j * BC + o_i)[None, :], mask=m_ij, other=0.0)
                        b_dq2 = tl.dot(b_dAqk.to(tl.float32), b_kg.to(tl.float32), b_dq2, allow_tf32=False)
                        b_dk2 = tl.dot(b_dAkk.to(tl.float32), b_kg.to(tl.float32), b_dk2, allow_tf32=False)
                    b_gqn = exp2(b_g - b_gn)
                    b_dq2 *= b_gqn
                    b_dk2 *= b_gqn

                # query-side diagonal. anchor is the midpoint gate of this BC tile.
                if tl.max(tl.max(b_g, 0) - tl.min(b_g, 0), 0) < 80.:
                    i_gm = i_ti + min(BC // 2, T - i_ti - 1)
                    b_gm = ((tl.min(b_g, 0) + tl.max(b_g, 0)) * .5)[None, :]
                    o_c = i_ti + o_i
                    m_c = o_c < T
                    m_dAf = m_c[:, None] & ((i_i * BC + o_i)[None, :] < BT)
                    b_dAqk_d = tl.load(p_dAqk + o_c[:, None] * (HV * BT) +
                                       (i_i * BC + o_i)[None, :], mask=m_dAf, other=0.0).to(tl.float32)
                    b_dAkk_d = tl.load(p_dAkk + o_c[:, None] * (HV * BT) +
                                       (i_i * BC + o_i)[None, :], mask=m_dAf, other=0.0).to(tl.float32)
                    m_i_d = (o_i[:, None] >= o_i[None, :]) & m_c[:, None] & m_c[None, :]
                    m_j_d = m_c[:, None]
                    b_dAqk_d = tl.where(m_i_d, b_dAqk_d, 0.)
                    b_dAkk_d = tl.where(m_i_d, b_dAkk_d, 0.)
                    b_g_d = tl.where(m_j_d, b_g - b_gm, 0.)
                    b_gq = tl.where(m_j_d, exp2(b_g_d), 0.)
                    b_gk = tl.where(m_j_d, exp2(-b_g_d), 0.)
                    b_k_exp = b_k.to(tl.float32) * b_gk
                    b_dq2 += tl.dot(b_dAqk_d, b_k_exp, allow_tf32=False) * b_gq
                    b_dk2 += tl.dot(b_dAkk_d, b_k_exp, allow_tf32=False) * b_gq
                else:
                    # pairwise per-column diag for unbounded (non-safe) gates
                    o_dA_c = i_ti * (HV * BT) + o_i * (HV * BT) + i_i * BC
                    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
                        b_dAqk_j = tl.load(p_dAqk + o_dA_c + j, mask=m_row, other=0.0)
                        b_dAkk_j = tl.load(p_dAkk + o_dA_c + j, mask=m_row, other=0.0)
                        b_kj = tl.load(p_k + i_ti * (H * K) + j * (H * K) + o_k, mask=m_k, other=0).to(tl.float32)
                        b_gkj = tl.load(p_g + i_ti * (HV * K) + j * (HV * K) + o_k, mask=m_k, other=0).to(tl.float32)
                        m_i = o_i[:, None] >= j
                        b_gqk = exp2(b_g - b_gkj[None, :])
                        b_dq2 += tl.where(m_i, b_dAqk_j[:, None] * b_kj[None, :] * b_gqk, 0.)
                        b_dk2 += tl.where(m_i, b_dAkk_j[:, None] * b_kj[None, :] * b_gqk, 0.)

                # ---- first-half outputs: dq2/db (past + diag contributions) ----
                b_db = tl.sum(b_dk2 * b_k.to(tl.float32), 1)
                b_dk2 = b_dk2 * b_b.to(tl.float32)[:, None]
                b_dg2 = b_q.to(tl.float32) * b_dq2
                b_dq2 += tl.load(p_dq + i_ti * (HV * K) + o_i[:, None] * (HV * K) +
                                 o_k[None, :], mask=m_ik, other=0.0).to(tl.float32)
                tl.store(p_dq2 + i_ti * (HV * K) + o_i[:, None] * (HV * K) +
                         o_k[None, :], b_dq2.to(dq2.dtype.element_ty), mask=m_ik)
                tl.store(p_db + i_ti * HV + o_i * HV, b_db.to(tl.float32), mask=m_row)

                # ---- future blocks (j > i) and diag-kk: dkt contribution ----
                b_dkt = tl.zeros([BC, BK], dtype=tl.float32)
                if i_i < NC_LOC - 1:
                    b_gn_f = tl.load(p_g + (min(i_ti + BC, T) - 1) * HV * K + o_k, mask=m_k, other=0.0).to(tl.float32)[None, :]
                    for i_j in range(i_i + 1, NC_LOC):
                        i_tj = tl.cast(i_t * BT + i_j * BC, tl.int64)
                        m_rowj = (i_tj + o_i) < T
                        m_ikj = m_rowj[:, None] & m_k[None, :]
                        b_qf = tl.load(p_q + i_tj * (H * K) + o_i[:, None] * (H * K) + o_k[None, :], mask=m_ikj, other=0.0)
                        b_kf = tl.load(p_k + i_tj * (H * K) + o_i[:, None] * (H * K) + o_k[None, :], mask=m_ikj, other=0.0)
                        b_gkf = tl.load(p_g + i_tj * (HV * K) + o_i[:, None] * (HV * K) +
                                        o_k[None, :], mask=m_ikj, other=0.0).to(tl.float32)
                        b_bf = tl.load(p_beta + i_tj * HV + o_i * HV, mask=m_rowj, other=0.0)
                        # transposed dA tiles: element (a, b) at dA + a + b*(HV*BT)
                        o_col = i_tj + o_i
                        m_t = (o_col[None, :] < T) & (o_row[:, None] < BT)
                        b_dAqk_f = tl.load(p_dAqk + i_tj * (HV * BT) +
                                           o_row[:, None] + o_i[None, :] * (HV * BT), mask=m_t, other=0.0)
                        b_dAkk_f = tl.load(p_dAkk + i_tj * (HV * BT) +
                                           o_row[:, None] + o_i[None, :] * (HV * BT), mask=m_t, other=0.0)
                        b_gkn = exp2(b_gkf - b_gn_f)
                        b_qg = b_qf * tl.where(m_rowj[:, None], b_gkn, 0)
                        b_kbg = b_kf * b_bf.to(tl.float32)[:, None] * tl.where(m_rowj[:, None], b_gkn, 0)
                        b_dkt = tl.dot(b_dAqk_f.to(tl.float32), b_qg.to(tl.float32), b_dkt, allow_tf32=False)
                        b_dkt = tl.dot(b_dAkk_f.to(tl.float32), b_kbg.to(tl.float32), b_dkt, allow_tf32=False)
                    b_dkt *= exp2(b_gn_f - b_g)

                # future-side diagonal. same midpoint anchor; dA rows are key offsets.
                if tl.max(tl.max(b_g, 0) - tl.min(b_g, 0), 0) < 80.:
                    i_gm = i_ti + min(BC // 2, T - i_ti - 1)
                    b_gm = ((tl.min(b_g, 0) + tl.max(b_g, 0)) * .5)[None, :]
                    o_c = i_ti + o_i
                    m_c = o_c < T
                    m_dAt = (o_i[:, None] < BT) & m_c[None, :]
                    b_dAqk_t = tl.load(p_dAqk + o_row[:, None] + o_c[None, :] * (HV * BT),
                                       mask=m_dAt, other=0.0).to(tl.float32)
                    b_dAkk_t = tl.load(p_dAkk + o_row[:, None] + o_c[None, :] * (HV * BT),
                                       mask=m_dAt, other=0.0).to(tl.float32)
                    m_i_t = (o_i[:, None] <= o_i[None, :]) & m_c[:, None] & m_c[None, :]
                    m_j_t = m_c[:, None]
                    b_dAqk_t = tl.where(m_i_t, b_dAqk_t, 0.)
                    b_dAkk_t = tl.where(m_i_t, b_dAkk_t, 0.)
                    b_g_d = tl.where(m_j_t, b_g - b_gm, 0.)
                    b_gq = tl.where(m_j_t, exp2(b_g_d), 0.)
                    b_gk = tl.where(m_j_t, exp2(-b_g_d), 0.)
                    b_q_exp = b_q.to(tl.float32) * b_gq
                    b_kb_exp = b_k.to(tl.float32) * b_b.to(tl.float32)[:, None] * b_gq
                    b_dkt += tl.dot(b_dAqk_t, b_q_exp, allow_tf32=False) * b_gk
                    b_dkt += tl.dot(b_dAkk_t, b_kb_exp, allow_tf32=False) * b_gk
                else:
                    # pairwise per-column diag (future side) for unbounded gates
                    for j in range(0, min(BC, T - i_t * BT - i_i * BC)):
                        b_dAqk_j = tl.load(p_dAqk + i_ti * (HV * BT) + j * (HV * BT) + i_i * BC + o_i)
                        b_dAkk_j = tl.load(p_dAkk + i_ti * (HV * BT) + j * (HV * BT) + i_i * BC + o_i)
                        b_qj = tl.load(p_q + i_ti * (H * K) + j * (H * K) + o_k, mask=m_k, other=0).to(tl.float32)
                        b_kbj = tl.load(p_k + i_ti * (H * K) + j * (H * K) + o_k, mask=m_k,
                                        other=0).to(tl.float32) * tl.load(p_beta + (i_ti + j) * HV)
                        b_gkj = tl.load(p_g + i_ti * (HV * K) + j * (HV * K) + o_k, mask=m_k, other=0).to(tl.float32)
                        m_i = o_i[:, None] <= j
                        b_gkq = exp2(b_gkj[None, :] - b_g)
                        b_dkt += tl.where(m_i, b_dAqk_j[:, None] * b_qj[None, :] * b_gkq, 0.)
                        b_dkt += tl.where(m_i, b_dAkk_j[:, None] * b_kbj[None, :] * b_gkq, 0.)

                # ---- second-half outputs: dk2/dg2 (adds future/dkt contributions) ----
                b_dg2 += (b_dk2 - b_dkt) * b_k.to(tl.float32) + tl.load(
                    p_dg + i_ti * (HV * K) + o_i[:, None] * (HV * K) + o_k[None, :], mask=m_ik, other=0.0)
                b_dk2 += tl.load(p_dk + i_ti * (HV * K) + o_i[:, None] * (HV * K) +
                                 o_k[None, :], mask=m_ik, other=0.0).to(tl.float32)
                b_dk2 += b_dkt
                tl.store(p_dk2 + i_ti * (HV * K) + o_i[:, None] * (HV * K) +
                         o_k[None, :], b_dk2.to(dk2.dtype.element_ty), mask=m_ik)
                tl.store(p_dg2 + i_ti * (HV * K) + o_i[:, None] * (HV * K) + o_k[None, :], b_dg2, mask=m_ik)

@input_guard
def chunk_kda_bwd_intra_npu(
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
    BK = min(64, triton.next_power_of_2(K))
    NK = triton.cdiv(K, BK)
    # Fixed 32x64 FP32 tiles fit the 192 KiB UB; 32x128 overflowed it.
    # Two K-slices own separate FP32 beta partials, reduced only after the kernel.
    BC = min(32, BT)
    if K != 128 or H != HV or chunk_size != 64 or cu_seqlens is not None or use_graph or safe_gate:
        raise ValueError("9382 matrix intra requires K128, equal heads, C64 and fixed-length non-safe-gate execution")
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    if use_graph:
        dq2 = get_static_buffer("intra_dq2", tuple(dq.shape), dq.dtype, dq.device, zero=True)
        dk2 = get_static_buffer("intra_dk2", tuple(dk.shape), dk.dtype, dk.device, zero=True)
        # Each K-slice owns one FP32 beta partial slab.
        db2 = get_static_buffer("intra_db2", (NK, *beta.shape), torch.float, beta.device, zero=True)
        dg2 = get_static_buffer("intra_dg2", tuple(dg.shape), torch.float, dg.device, zero=True)
    else:
        dq2 = torch.empty_like(dq)
        dk2 = torch.empty_like(dk)
        db2 = beta.new_empty(NK, *beta.shape, dtype=torch.float)
        dg2 = torch.empty_like(dg, dtype=torch.float)
    num_core = get_npu_properties()['num_aicore']
    chunk_kda_bwd_kernel_intra_npu[(num_core,)](
        NT_TOTAL=NT,
        q=q, k=k, g=g, beta=beta, dAqk=dAqk, dAkk=dAkk,
        dq=dq, dq2=dq2, dk=dk, dk2=dk2, dg=dg, dg2=dg2, db=db2,
        cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
        B=B, T=T, H=H, HV=HV, K=K, BT=BT, BC=BC, BK=BK,
        SAFE_GATE=safe_gate,
        USE_GRAPH=use_graph,
        **ascend_compile_kwargs(),
    )
    return dq2, dk2, db2.sum(0).add_(db), dg2

kda_intra=chunk_kda_bwd_intra_npu
