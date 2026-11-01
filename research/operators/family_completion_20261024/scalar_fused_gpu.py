"""Single-program scalar-gate residual WY, keeping P and dM on chip."""
from common import torch,triton
import triton.language as tl

@triton.jit
def kernel(K0,V0,G,BETA,A,DR,R,H0,DK,DV,DB,DG,CU,CI,
           T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
           C:tl.constexpr,NK:tl.constexpr,NV:tl.constexpr,GATED:tl.constexpr,GRAM:tl.constexpr,
           VAR:tl.constexpr,SVF:tl.constexpr):
    it=tl.program_id(0);bh=tl.program_id(1);ih=bh%HV
    if VAR:
        tg=it;seq=tl.load(CI+2*it);it=tl.load(CI+2*it+1);bos=tl.load(CU+seq);length=tl.load(CU+seq+1)-bos
    else:bos=(bh//HV)*T;length=T;tg=(bh//HV)*tl.cdiv(T,C)+it
    rr=tl.arange(0,C);tt=it*C+rr;off=(bos+tt)*HV+ih;ko=(bos+tt)*H+ih//(HV//H);vv=tl.arange(0,NV)
    pair=(tt[:,None]<length)&(tt[None,:]<length);vm=(tt[:,None]<length)&(vv[None,:]<V)
    at=tl.load(A+off[None,:]*C+rr[:,None],pair,0)
    dr=tl.load(DR+off[:,None]*V+vv[None,:],vm,0)
    p=tl.dot(at,dr,allow_tf32=False)
    r=tl.load(R+off[:,None]*V+vv[None,:],vm,0)
    dm=-tl.dot(p.to(r.dtype),tl.trans(r),allow_tf32=False)
    value=tl.load(V0+off[:,None]*V+vv[None,:],vm,0)
    beta=tl.load(BETA+off,tt<length,0).to(tl.float32)
    db=tl.sum(p*value,1);dg=tl.zeros((C,),tl.float32)
    tl.store(DV+off[:,None]*V+vv[None,:],p*beta[:,None],vm)
    if GATED:
        g=tl.load(G+off,tt<length,0).to(tl.float32);eg=tl.exp2(g)
        dm=dm*tl.exp2(tl.where(rr[:,None]>rr[None,:],g[:,None]-g[None,:],0))
    else:eg=tl.full((C,),1,tl.float32)
    dm=tl.where(pair&(rr[:,None]>rr[None,:]),dm,0).to(K0.dtype.element_ty)
    for ik in range(tl.cdiv(K,NK)):
        kk=ik*NK+tl.arange(0,NK);km=(tt[:,None]<length)&(kk[None,:]<K)
        key=tl.load(K0+ko[:,None]*K+kk[None,:],km,0)
        if SVF:hp=(tg*HV+ih)*K*V+vv[:,None]*K+kk[None,:]
        else:hp=(tg*HV+ih)*K*V+kk[None,:]*V+vv[:,None]
        hs=tl.load(H0+hp,(vv[:,None]<V)&(kk[None,:]<K),0).to(key.dtype)
        de=-tl.dot(p.to(key.dtype),hs,allow_tf32=False)
        left=tl.dot(dm,key,allow_tf32=False)
        right=tl.dot(tl.trans(dm),(key*beta[:,None]).to(key.dtype),allow_tf32=False)
        dk=de*(beta*eg)[:,None]+left*beta[:,None]+right
        db+=tl.sum(de*key*eg[:,None]+left*key,1)
        if GATED:
            dg+=tl.sum(de*key*(beta*eg)[:,None],1)
            if GRAM:
                gram=tl.dot(key,tl.trans(key),allow_tf32=False)
                j=gram*dm.to(tl.float32)*beta[:,None]
                dg+=tl.sum(j,1)-tl.sum(j,0)
            else:dg+=tl.sum(key*(left*beta[:,None]-right),1)
        tl.store(DK+off[:,None]*K+kk[None,:],dk,km)
    tl.store(DB+off,db,tt<length)
    if GATED:tl.store(DG+off,dg,tt<length)

def wy(k,v,beta,A,du,h,r,g=None,gram=False,state_v_first=False,cu_seqlens=None,chunk_indices=None,**unused):
    B,T,H,K=k.shape;HV=v.shape[2];V=v.shape[-1];C=A.shape[-1]
    from fla.ops.utils import prepare_chunk_indices
    if cu_seqlens is not None and chunk_indices is None:chunk_indices=prepare_chunk_indices(cu_seqlens,C)
    nt=len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T,C)
    dk=torch.empty((B,T,HV,K),device=k.device,dtype=k.dtype);dv=torch.empty_like(v);db=torch.empty_like(beta);dg=torch.empty_like(g) if g is not None else None
    kernel[(nt,B*HV)](k,v,g,beta,A,du,r,h,dk,dv,db,dg,cu_seqlens,chunk_indices,T=T,H=H,HV=HV,K=K,V=V,C=C,NK=32,NV=triton.next_power_of_2(V),GATED=g is not None,GRAM=gram,VAR=cu_seqlens is not None,SVF=state_v_first,num_warps=8,num_stages=2)
    if H!=HV:dk=dk.reshape(B,T,H,HV//H,K).sum(3)
    return (dk,dv,db,dg) if g is not None else (dk,dv,db)
