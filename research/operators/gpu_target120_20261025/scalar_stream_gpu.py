"""One-CTA chunk adjoint with streamed V operands and input-dtype new P.

No new algebra; D/S remain on chip. Fixed NK32/NV64/4warps/1stage.
FP32 P computes DV/db_value before only its new shared storage is converted.
"""
from common import torch,triton
import triton.language as tl

@triton.jit
def kernel(Q,K0,V0,G,BETA,A,DR,R,H0,DH,DO,P,DQ,DK,DV,DB,DG,CU,CI,
           T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
           C:tl.constexpr,NK:tl.constexpr,NV:tl.constexpr,SCALE:tl.constexpr,
           GATED:tl.constexpr,VAR:tl.constexpr,SVF:tl.constexpr):
    it=tl.program_id(0);bh=tl.program_id(1);ih=bh%HV
    if VAR:
        tg=it;seq=tl.load(CI+2*it);it=tl.load(CI+2*it+1)
        bos=tl.load(CU+seq);length=tl.load(CU+seq+1)-bos
    else:bos=(bh//HV)*T;length=T;tg=(bh//HV)*tl.cdiv(T,C)+it
    rr=tl.arange(0,C);tt=it*C+rr;off=(bos+tt)*HV+ih
    ko=(bos+tt)*H+ih//(HV//H);valid=tt<length;pair=valid[:,None]&valid[None,:]
    beta=tl.load(BETA+off,valid,0).to(tl.float32)
    dm=tl.zeros((C,C),tl.float32);ds=tl.zeros((C,C),tl.float32);db=tl.zeros((C,),tl.float32)
    at=tl.load(A+off[None,:]*C+rr[:,None],pair,0)
    for iv in range(tl.cdiv(V,NV)):
        vv=iv*NV+tl.arange(0,NV);vm=valid[:,None]&(vv[None,:]<V)
        dr=tl.load(DR+off[:,None]*V+vv[None,:],vm,0)
        p=tl.dot(at,dr,allow_tf32=False)
        r=tl.load(R+off[:,None]*V+vv[None,:],vm,0)
        dout=tl.load(DO+off[:,None]*V+vv[None,:],vm,0)
        dm-=tl.dot(p.to(r.dtype),tl.trans(r),allow_tf32=False)
        ds+=tl.dot(dout,tl.trans(r),allow_tf32=False)
        value=tl.load(V0+off[:,None]*V+vv[None,:],vm,0)
        db+=tl.sum(p*value,1)
        tl.store(DV+off[:,None]*V+vv[None,:],p*beta[:,None],vm)
        tl.store(P+off[:,None]*V+vv[None,:],p,vm)
    # P is exclusively produced/consumed by this CTA. Barrier orders its
    # global writes before potentially different warp layouts reload them.
    tl.debug_barrier()
    dg=tl.zeros((C,),tl.float32);dg_last=tl.full((),0,tl.float32)
    if GATED:
        g=tl.load(G+off,valid,0).to(tl.float32)
        gn=tl.load(G+(bos+tl.minimum((it+1)*C,length)-1)*HV+ih).to(tl.float32)
        eg=tl.exp2(g);en=tl.exp2(gn);er=tl.where(valid,tl.exp2(gn-g),0)
        fac=tl.exp2(tl.where(pair&(rr[:,None]>=rr[None,:]),g[:,None]-g[None,:],0))
        dm*=fac;ds*=fac
    else:eg=tl.full((C,),1,tl.float32);er=tl.full((C,),1,tl.float32)
    dm=tl.where(pair&(rr[:,None]>rr[None,:]),dm,0).to(K0.dtype.element_ty)
    ds=tl.where(pair&(rr[:,None]>=rr[None,:]),ds*SCALE,0).to(K0.dtype.element_ty)
    for ik in range(tl.cdiv(K,NK)):
        kk=ik*NK+tl.arange(0,NK);km=valid[:,None]&(kk[None,:]<K)
        dq0=tl.zeros((C,NK),tl.float32);dk0=tl.zeros((C,NK),tl.float32);de=tl.zeros((C,NK),tl.float32)
        for iv in range(tl.cdiv(V,NV)):
            vv=iv*NV+tl.arange(0,NV);vm=valid[:,None]&(vv[None,:]<V)
            if SVF:hp=(tg*HV+ih)*K*V+vv[:,None]*K+kk[None,:]
            else:hp=(tg*HV+ih)*K*V+kk[None,:]*V+vv[:,None]
            hm=(vv[:,None]<V)&(kk[None,:]<K)
            hs=tl.load(H0+hp,hm,0);dhs=tl.load(DH+hp,hm,0)
            if GATED:dg_last+=tl.sum(hs*dhs)*en
            dout=tl.load(DO+off[:,None]*V+vv[None,:],vm,0)
            dq0=tl.dot(dout,hs.to(dout.dtype),dq0,allow_tf32=False)
            r=tl.load(R+off[:,None]*V+vv[None,:],vm,0)
            dk0=tl.dot(r,dhs.to(r.dtype),dk0,allow_tf32=False)
            pp=tl.load(P+off[:,None]*V+vv[None,:],vm,0)
            de-=tl.dot(pp,hs.to(pp.dtype),allow_tf32=False)
        key=tl.load(K0+ko[:,None]*K+kk[None,:],km,0)
        query=tl.load(Q+ko[:,None]*K+kk[None,:],km,0)
        dq=dq0*(eg*SCALE)[:,None]+tl.dot(ds,key,allow_tf32=False)
        tl.store(DQ+off[:,None]*K+kk[None,:],dq,km)
        dk_state=dk0*er[:,None]
        dk_direct=dk_state+tl.dot(tl.trans(ds),query,allow_tf32=False)
        left=tl.dot(dm,key,allow_tf32=False)
        right=tl.dot(tl.trans(dm),(key*beta[:,None]).to(key.dtype),allow_tf32=False)
        dk=dk_direct+(de*eg[:,None]+left)*beta[:,None]+right
        db+=tl.sum((de*eg[:,None]+left)*key,1)
        if GATED:
            dg+=tl.sum(dq*query-dk_direct*key+de*key*(beta*eg)[:,None]+key*(left*beta[:,None]-right),1)
            dg_last+=tl.sum(dk_state*key)
        tl.store(DK+off[:,None]*K+kk[None,:],dk,km)
    tl.store(DB+off,db,valid)
    if GATED:
        dg=tl.where(tt==tl.minimum((it+1)*C,length)-1,dg+dg_last,dg)
        dg=tl.cumsum(tl.where(valid,dg,0),0,reverse=True)
        tl.store(DG+off,dg,valid)

def adjoint(q,k,v,g,beta,A,dr,r,h,dh,do,scale,state_v_first=False,
            cu_seqlens=None,chunk_indices=None):
    from fla.ops.utils import prepare_chunk_indices
    B,T,H,K=k.shape;HV=v.shape[2];V=v.shape[-1];C=A.shape[-1]
    if cu_seqlens is not None and chunk_indices is None:chunk_indices=prepare_chunk_indices(cu_seqlens,C)
    nt=len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T,C)
    p=torch.empty_like(v);dq=torch.empty((B,T,HV,K),device=q.device,dtype=q.dtype);dk=torch.empty_like(dq)
    dv=torch.empty_like(v);db=torch.empty_like(beta);dg=torch.empty_like(g) if g is not None else None
    kernel[(nt,B*HV)](q,k,v,g,beta,A,dr,r,h,dh,do,p,dq,dk,dv,db,dg,cu_seqlens,chunk_indices,
        T=T,H=H,HV=HV,K=K,V=V,C=C,NK=32,NV=64,SCALE=scale,GATED=g is not None,
        VAR=cu_seqlens is not None,SVF=state_v_first,num_warps=4,num_stages=1)
    if H!=HV:dq=dq.reshape(B,T,H,HV//H,K).sum(3);dk=dk.reshape(B,T,H,HV//H,K).sum(3)
    return dq,dk,dv,dg,db
