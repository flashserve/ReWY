"""Fixed multi-stage control using the unchanged frozen reference JIT body.

Used only to isolate the full-K rewrite at identical BV/warps/stages.
The headline 'original' baseline keeps its original autotuner unchanged.
"""
from common import torch,triton
from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64

reference_kernel=chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64
while not isinstance(reference_kernel,triton.runtime.jit.JITFunction):
    reference_kernel=reference_kernel.fn


def state_bwd(q,k,w,do,dv,g=None,gk=None,h0=None,dht=None,scale=None,
              state_v_first=False,cu_seqlens=None,chunk_size=64,
              chunk_indices=None,chunk_offsets=None,stages=3,value_block=32,**kw):
    from fla.ops.utils import prepare_chunk_indices,prepare_chunk_offsets
    assert gk is None
    B,T,H,K=q.shape;V=do.shape[-1];HV=do.shape[2];C=chunk_size
    if cu_seqlens is None:N=B;nt=triton.cdiv(T,C)
    else:
        N=len(cu_seqlens)-1
        if chunk_indices is None:chunk_indices=prepare_chunk_indices(cu_seqlens,C)
        nt=len(chunk_indices)
        if chunk_offsets is None:chunk_offsets=prepare_chunk_offsets(cu_seqlens,C)
    dims=(V,K) if state_v_first else (K,V)
    dh=torch.empty((B,nt,HV,*dims),device=q.device,dtype=q.dtype)
    dh0=torch.empty_like(h0,dtype=torch.float32) if h0 is not None else None
    dr=torch.empty_like(dv)
    reference_kernel[(triton.cdiv(V,value_block)*N*HV,)](
        q=q,k=k,w=w,g=g,gk=gk,dht=dht,dh0=dh0,do=do,dh=dh,dv=dv,dv2=dr,
        cu_seqlens=cu_seqlens,chunk_offsets=chunk_offsets,scale=scale,
        T=T,H=H,HV=HV,K=K,V=V,BT=C,BV=value_block,USE_G=g is not None,USE_GK=False,
        USE_INITIAL_STATE=h0 is not None,USE_FINAL_STATE_GRADIENT=dht is not None,
        STATE_V_FIRST=state_v_first,IS_VARLEN=cu_seqlens is not None,
        num_warps=4,num_stages=stages)
    return dh,dh0,dr
