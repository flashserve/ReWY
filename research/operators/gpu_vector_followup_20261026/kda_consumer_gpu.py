"""P/dT producer and fused state+intra consumer; full KDA backward interface."""
import os
import torch
import triton
from fla.ops.utils import prepare_chunk_indices,prepare_chunk_offsets
from fla.utils import IS_NPU,ascend_compile_kwargs
from kda_consumer_kernel_gpu import fused_kernel


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
