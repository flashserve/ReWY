"""Backward-only native residual producer; unchanged forward and v2 fallback.

This module is an internal experiment hook, not a new public KDA API.
"""
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
import torch
import triton
from fla.utils import ascend_compile_kwargs, input_guard
from kernels import kda_residual_npu_onchip as v2

_counts = Counter()
_loaded = False
_workspace = {}
_library = Path(__file__).parent / 'native/build/libkda_residual_native.so'
_variant = 'producer'
_old_intra = None
_pending_de = {}
_last_flags = None
_intra_tile = None


def dispatch_stats(reset=False):
    result = {'native': _counts['native'], 'fallback': _counts['fallback'],
              'fallback_reasons': {k[9:]: v for k, v in _counts.items() if k.startswith('fallback:')}}
    result.update({k: _counts[k] for k in ('producer_native','consumer_native','de_producer','de_consumer')})
    result.update(variant=_variant, producer_backend='triton_de' if _variant=='joint' else ('ascendc' if _variant=='producer' else 'frozen_v2'),
                  consumer_backend='ascendc_matrix_bf16' if _variant!='producer' else 'frozen_intra')
    if _last_flags is not None:
        result['pairwise_tasks_last_call'] = int(_last_flags[::8].cpu().sum())
    if _intra_tile is not None:
        result['intra_tile_m'],result['intra_tile_n'],result['intra_explicit_ub_bytes']=_intra_tile[:3]
        result['intra_anchor']='global' if _intra_tile[3] else 'channel'
    if reset:
        _counts.clear()
    return result


def _load(device):
    global _loaded,_intra_tile
    if not _loaded:
        if not _library.exists():
            raise RuntimeError(f'Native library not built: {_library}')
        torch.ops.load_library(str(_library))
        _loaded = True
        _intra_tile=tuple(torch.ops.kda_residual_native.intra_config())
    key = (str(device), int(torch.npu.current_stream(device).npu_stream))
    if key not in _workspace:
        # MatmulClient system control/workspace crosses AIC/AIV execution roles.
        # It is scratch, not a saved forward activation, and is reused across calls.
        _workspace[key] = torch.empty(16 * 1024 * 1024, dtype=torch.uint8, device=device)
    return _workspace[key]


@input_guard
def native_wy(q, k, v, v_new, g, beta, A, h, do, dh, dv,
              scale=None, state_v_first=False, cu_seqlens=None,
              chunk_size=64, chunk_indices=None, chunk_offsets=None, use_graph=False):
    B, T, H, K = k.shape
    HV, V = v.shape[2:]
    reason = None
    if cu_seqlens is not None:
        reason = 'varlen'
    elif chunk_size != 64 or T % 64:
        reason = 'chunk_or_tail'
    elif K != 128 or V != 128:
        reason = 'feature_dimension'
    elif H != HV:
        reason = 'grouped_value_heads'
    elif any(x.dtype != torch.bfloat16 for x in (k, v, v_new, beta, A, dv, h, dh)):
        reason = 'input_dtype'
    elif g.dtype != torch.float32:
        reason = 'state_gate_dtype'
    if reason:
        _counts['fallback'] += 1
        _counts['fallback:' + reason] += 1
        old = v2._CORE_BK
        try:
            v2._CORE_BK = 128
            return v2.chunk_kda_bwd_wy_dqkg_fused_npu(
                q, k, v, v_new, g, beta, A, h, do, dh, dv,
                scale, state_v_first, cu_seqlens, chunk_size,
                chunk_indices, chunk_offsets, use_graph)
        finally:
            v2._CORE_BK = old

    workspace = _load(v.device)
    _counts['native'] += 1
    _counts['producer_native'] += 1
    dq = torch.empty((B, T, HV, K), device=g.device, dtype=torch.float32)
    dk = torch.empty_like(dq)
    dg = torch.empty_like(g, dtype=torch.float32)
    dv_out = torch.empty_like(v)
    db = torch.empty_like(beta, dtype=torch.float32)
    dm = torch.empty_like(A, dtype=torch.float32)
    # Existing state branch requires this layout. Producer-only removes the five
    # value-side full tensor copies; state copies remain included in backward.
    if state_v_first:
        h = v2._k_major_state(h, 'native_h_kmajor', use_graph)
        dh = v2._k_major_state(dh, 'native_dh_kmajor', use_graph)
    bk, bv = v2._get_bk(K), v2._get_bv(V)
    cores = v2.get_npu_properties()['num_vectorcore']
    if chunk_offsets is None:
        chunk_offsets = g.new_zeros(1, dtype=torch.int64)
    for key_tile in range(triton.cdiv(K, bk)):
        v2.chunk_kda_bwd_kernel_wy_k_part_npu[(cores,)](
            q=q, k=k, v_new=v_new, g=g, h=h, do=do, dh=dh,
            dq=dq, dk=dk, dg=dg, cu_seqlens=None, chunk_indices=None,
            chunk_offsets=chunk_offsets, scale=scale, T=T, BH=B * HV,
            task_num=(T // 64) * B * HV, num_core=cores, H=H, HV=HV,
            K=K, V=V, BT=64, BC=v2._BC, BK=bk, BV=bv,
            STATE_V_FIRST=False, IS_VARLEN=False, USE_GRAPH=use_graph,
            K_OFFSET=key_tile, **ascend_compile_kwargs())
    torch.ops.kda_residual_native.producer(v, v_new, A, dv, h, k, g, beta,
                                          dk, dg, dv_out, db, dm, workspace)
    return dq, dk, dv_out, db, dg, dm


def _consumer_reason(q,k,g,beta,dAqk,dAkk,dq,dk,dg,cu_seqlens,chunk_size,safe_gate):
    if cu_seqlens is not None: return 'varlen'
    if chunk_size!=64 or k.shape[1]%64: return 'chunk_or_tail'
    if k.shape[-1]!=128 or k.shape!=g.shape or q.shape!=k.shape: return 'feature_or_grouped_heads'
    if safe_gate: return 'safe_gate_transform'
    if any(x.dtype!=torch.bfloat16 for x in (q,k,beta)): return 'input_dtype'
    if any(x.dtype!=torch.float32 for x in (g,dAqk,dAkk,dq,dk,dg)): return 'accumulator_dtype'
    return None


def _stream_key(x):
    return (str(x.device),int(torch.npu.current_stream(x.device).npu_stream),x.data_ptr())


@input_guard
def native_intra(q,k,g,beta,dAqk,dAkk,dq,dk,db,dg,cu_seqlens=None,chunk_indices=None,
                 chunk_size=64,safe_gate=False,lower_bound=None,use_graph=False):
    global _last_flags
    reason=_consumer_reason(q,k,g,beta,dAqk,dAkk,dq,dk,dg,cu_seqlens,chunk_size,safe_gate)
    if reason:
        if _stream_key(dAkk) in _pending_de:
            raise RuntimeError('DE was produced but native consumer cannot execute: '+reason)
        _counts['fallback']+=1;_counts['fallback:'+reason]+=1
        return _old_intra(q=q,k=k,g=g,beta=beta,dAqk=dAqk,dAkk=dAkk,dq=dq,dk=dk,db=db,dg=dg,
            cu_seqlens=cu_seqlens,chunk_indices=chunk_indices,chunk_size=chunk_size,safe_gate=safe_gate,lower_bound=lower_bound,use_graph=use_graph)
    B,T,H,K=k.shape
    de=_pending_de.pop(_stream_key(dAkk),None)
    joint=de is not None
    workspace=_load(k.device)
    tile_m,tile_n=_intra_tile[:2]
    out=[torch.empty_like(x) for x in (dq,dk,dg)]
    parts=torch.empty((K//tile_n,B,T,H),device=k.device,dtype=torch.float32)
    flags=torch.empty((B*T//64*H*(64//tile_m)*(K//tile_n)*8,),device=k.device,dtype=torch.int32)
    empty=de if joint else torch.empty(0,device=k.device)
    torch.ops.kda_residual_native.intra(q,k,g,beta,dAqk,dAkk,dq,dk,dg,empty,*out,parts,flags,workspace,joint)
    _last_flags=flags
    _counts['native']+=1;_counts['consumer_native']+=1
    if joint:_counts['de_consumer']+=1
    return out[0],out[1],parts.sum(0).add_(db),out[2]


class _DEKernel:
    def __getitem__(self,grid):
        def launch(**kwargs):
            from native.de_core import residual_de_core
            de=torch.empty_like(kwargs['dk'])
            _pending_de[_stream_key(kwargs['dA_acc'])]=de
            residual_de_core[grid](de_out=de,**kwargs)
            _counts['de_producer']+=1
        return launch


@input_guard
def joint_wy(q,k,v,v_new,g,beta,A,h,do,dh,dv,scale=None,state_v_first=False,
             cu_seqlens=None,chunk_size=64,chunk_indices=None,chunk_offsets=None,use_graph=False):
    # The v2 state and layout paths stay identical to stage A. Only the direct
    # gradient boundary changes: FP32 DE replaces producer dk/dG read-modify-write.
    supported=(cu_seqlens is None and chunk_size==64 and k.shape[1]%64==0 and
               k.shape[-1]==v.shape[-1]==128 and k.shape==g.shape and k.shape[2]==v.shape[2] and
               all(x.dtype==torch.bfloat16 for x in (q,k,v,v_new,beta,A,h,dh,dv)) and g.dtype==torch.float32)
    oldcore,oldbk=v2.residual_npu_core,v2._CORE_BK
    try:
        v2._CORE_BK=128
        if supported:v2.residual_npu_core=_DEKernel()
        return v2.chunk_kda_bwd_wy_dqkg_fused_npu(q,k,v,v_new,g,beta,A,h,do,dh,dv,scale,
            state_v_first,cu_seqlens,chunk_size,chunk_indices,chunk_offsets,use_graph)
    finally:
        v2.residual_npu_core=oldcore;v2._CORE_BK=oldbk
