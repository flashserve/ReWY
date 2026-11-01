"""Single-variable GPU tiling/warp probes using the exact old kernel body."""
import os
from common import torch,triton
from scalar_fused_gpu import kernel
LAST_CONFIG={}
def wy(k,v,beta,A,du,h,r,g=None,gram=False,state_v_first=False,cu_seqlens=None,chunk_indices=None,**unused):
    B,T,H,K=k.shape;HV=v.shape[2];V=v.shape[-1];C=A.shape[-1]
    from fla.ops.utils import prepare_chunk_indices
    if cu_seqlens is not None and chunk_indices is None:chunk_indices=prepare_chunk_indices(cu_seqlens,C)
    nt=len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T,C)
    dk=torch.empty((B,T,HV,K),device=k.device,dtype=k.dtype);dv=torch.empty_like(v);db=torch.empty_like(beta);dg=torch.empty_like(g) if g is not None else None
    compiled=kernel[(nt,B*HV)](k,v,g,beta,A,du,r,h,dk,dv,db,dg,cu_seqlens,chunk_indices,T=T,H=H,HV=HV,K=K,V=V,C=C,NK=int(os.environ.get("GDN_NK","32")),NV=triton.next_power_of_2(V),GATED=g is not None,GRAM=gram,VAR=cu_seqlens is not None,SVF=state_v_first,num_warps=int(os.environ.get("GDN_WARPS","4")),num_stages=int(os.environ.get("GDN_STAGES","2")))
    LAST_CONFIG.clear();LAST_CONFIG.update(n_regs=getattr(compiled,"n_regs",None),n_spills=getattr(compiled,"n_spills",None),shared_bytes=getattr(compiled.metadata,"shared",None),num_warps=compiled.metadata.num_warps,nk=int(os.environ.get("GDN_NK","32")),nv=triton.next_power_of_2(V))
    if H!=HV:dk=dk.reshape(B,T,H,HV//H,K).sum(3)
    return (dk,dv,db,dg) if g is not None else (dk,dv,db)
