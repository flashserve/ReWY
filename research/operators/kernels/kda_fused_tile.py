"""P/dT producer and fused state+intra consumer; full KDA backward interface."""
import os
import torch
import triton
from fla.ops.utils import prepare_chunk_indices,prepare_chunk_offsets
from fla.utils import IS_NPU,ascend_compile_kwargs
from .kda_residual_tri_beta import residual_tri_beta_producer
from .kda_joint_producer import residual_joint_producer
from .kda_fused_tile_kernel import fused_kernel
from .kda_full_producer import full_producer


def fused_wy(q,k,v,v_new,g,beta,A,h,do,dh,dv,scale=None,state_v_first=False,
             cu_seqlens=None,chunk_size=64,chunk_indices=None,chunk_offsets=None,use_graph=False,
             materialized=True,endpoint_state=None):
    assert not use_graph, 'research wrapper does not support static graph padding'
    B,T,H,K=k.shape
    HV,V=v.shape[2:]
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices=prepare_chunk_indices(cu_seqlens,chunk_size)
    NT=triton.cdiv(T,chunk_size) if chunk_indices is None else len(chunk_indices)
    dq=torch.empty((B,T,HV,K),device=q.device,dtype=torch.float32)
    dk=torch.empty_like(dq);dg=torch.empty_like(dq)
    db=torch.empty_like(beta,dtype=torch.float32)
    compensate=os.environ.get('KDA_COMPENSATE','0')=='1'
    logit_scale=float(os.environ.get('KDA_FUSED_LOGIT_SCALE','0'))
    assert not compensate or materialized
    pp=torch.empty_like(v,dtype=torch.float32 if compensate else v.dtype);dv2=torch.empty_like(v)
    dt=torch.empty_like(A,dtype=torch.float32) if materialized else pp
    if IS_NPU:
        db.zero_();dt.zero_()
    producer=residual_tri_beta_producer if materialized else residual_joint_producer
    kw=ascend_compile_kwargs() if IS_NPU else dict(num_warps=4,num_stages=2)
    if IS_NPU or compensate or logit_scale>0:
        full_producer[(NT,B*HV)](v,v_new,A,dv,beta,pp,dv2,db,dt,cu_seqlens,chunk_indices,T,
            HV=HV,V=V,VP=triton.next_power_of_2(V),BT=chunk_size,VARLEN=cu_seqlens is not None,
            MATERIALIZED=materialized,COMPENSATE=compensate,LOGIT_SCALE=logit_scale,**kw)
    else:
        producer[(NT*(chunk_size//32),B*HV)](
            v,v_new,A,dv,beta,pp,dv2,db,dt,cu_seqlens,chunk_indices,T,
            HV=HV,V=V,VP=triton.next_power_of_2(V),BT=chunk_size,BR=32,
            IS_VARLEN=cu_seqlens is not None,USE_GRAPH=False,ROW_FAST=True,**kw)
    return dq,dk,dv2,db,dg,(dt,pp,v_new,h,dh,do,scale,state_v_first,materialized,endpoint_state)


def fused_intra(q,k,g,beta,dAqk,dAkk,dq,dk,db,dg,cu_seqlens=None,chunk_indices=None,
                chunk_size=64,safe_gate=False,lower_bound=None,use_graph=False):
    dt,pp,R,h,dh,do,scale,vf,materialized,ht=dAkk
    B,T,H,K=k.shape
    HV,V=R.shape[2:]
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices=prepare_chunk_indices(cu_seqlens,chunk_size)
    offsets=prepare_chunk_offsets(cu_seqlens,chunk_size) if cu_seqlens is not None else None
    NT=triton.cdiv(T,chunk_size) if chunk_indices is None else len(chunk_indices)
    BK=int(os.environ.get('KDA_TILE_BK','16' if IS_NPU else '32'))
    kw=ascend_compile_kwargs() if IS_NPU else dict(num_warps=4,num_stages=2)
    fused_kernel[(NT*triton.cdiv(K,BK),B*HV)](
        q,k,g,beta,dAqk,dt,R,dq,dk,dg,db,cu_seqlens,chunk_indices,T,
        h,dh,do,pp,offsets,scale,ht if ht is not None else h,H=H,HV=HV,K=K,V=V,VP=triton.next_power_of_2(V),
        BT=chunk_size,BK=BK,VARLEN=cu_seqlens is not None,MATERIALIZED=materialized,
        BETA_COMPLETE=True,REVERSE_CUMSUM=True,DOT_PRECISION='ieee' if IS_NPU else 'tf32x3',
        LOWP=os.environ.get('KDA_INTRA_DOT','fp32')=='lowp',VF=vf,
        COMPENSATE=os.environ.get('KDA_COMPENSATE','0')=='1',
        LOGIT_SCALE=float(os.environ.get('KDA_FUSED_LOGIT_SCALE','0')),
        ENDPOINT=ht is not None,**kw)
    return dq,dk,db,dg
