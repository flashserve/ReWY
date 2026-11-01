"""Unified chunk adjoint: share residual/output contractions and state reads.

Keep the original contractions and precision with shorter live ranges.
One fixed schedule for all shapes: C rows, NK32, four warps, one stage.
No shape dispatch or benchmark-case lookup. No extra saved forward tensors.
"""
from ._common import torch, triton
import triton.language as tl


@triton.jit
def unified(Q,K0,V0,G,BETA,A,DR,R,H0,DH,DO,DQ,DK,DV,DB,DG,CU,CI,
            T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
            C:tl.constexpr,NK:tl.constexpr,NV:tl.constexpr,SCALE:tl.constexpr,
            GATED:tl.constexpr,VAR:tl.constexpr,SVF:tl.constexpr):
    it=tl.program_id(0);bh=tl.program_id(1);ih=bh%HV
    if VAR:
        tg=it;seq=tl.load(CI+2*it);it=tl.load(CI+2*it+1)
        bos=tl.load(CU+seq);length=tl.load(CU+seq+1)-bos
    else:bos=(bh//HV)*T;length=T;tg=(bh//HV)*tl.cdiv(T,C)+it
    rr=tl.arange(0,C);tt=it*C+rr;off=(bos+tt)*HV+ih
    ko=(bos+tt)*H+ih//(HV//H);vv=tl.arange(0,NV)
    valid=tt<length;pair=valid[:,None]&valid[None,:]
    vm=valid[:,None]&(vv[None,:]<V)
    at=tl.load(A+off[None,:]*C+rr[:,None],pair,0)
    dr=tl.load(DR+off[:,None]*V+vv[None,:],vm,0)
    p=tl.dot(at,dr,allow_tf32=False)
    r=tl.load(R+off[:,None]*V+vv[None,:],vm,0)
    dout=tl.load(DO+off[:,None]*V+vv[None,:],vm,0)
    dm=-tl.dot(p.to(r.dtype),tl.trans(r),allow_tf32=False)
    ds=tl.dot(dout,tl.trans(r),allow_tf32=False)
    value=tl.load(V0+off[:,None]*V+vv[None,:],vm,0)
    beta=tl.load(BETA+off,valid,0).to(tl.float32)
    db=tl.sum(p*value,1);dg=tl.zeros((C,),tl.float32)
    dg_last=tl.full((),0,tl.float32)
    tl.store(DV+off[:,None]*V+vv[None,:],p*beta[:,None],vm)
    if GATED:
        g=tl.load(G+off,valid,0).to(tl.float32)
        gn=tl.load(G+(bos+tl.minimum((it+1)*C,length)-1)*HV+ih).to(tl.float32)
        eg=tl.exp2(g);en=tl.exp2(gn);er=tl.where(valid,tl.exp2(gn-g),0)
        fac=tl.exp2(tl.where(pair&(rr[:,None]>=rr[None,:]),g[:,None]-g[None,:],0))
        dm=dm*fac;ds=ds*fac
    else:
        eg=tl.full((C,),1,tl.float32);er=tl.full((C,),1,tl.float32)
    dm=tl.where(pair&(rr[:,None]>rr[None,:]),dm,0).to(K0.dtype.element_ty)
    ds=tl.where(pair&(rr[:,None]>=rr[None,:]),ds*SCALE,0).to(K0.dtype.element_ty)
    # P no longer needs its FP32 value after DV and the value-beta reduction.
    # This is the same operand conversion as the old per-iteration dot cast.
    pp=p.to(K0.dtype.element_ty)
    for ik in range(tl.cdiv(K,NK)):
        kk=ik*NK+tl.arange(0,NK);km=valid[:,None]&(kk[None,:]<K)
        key=tl.load(K0+ko[:,None]*K+kk[None,:],km,0)
        query=tl.load(Q+ko[:,None]*K+kk[None,:],km,0)
        if SVF:hp=(tg*HV+ih)*K*V+vv[:,None]*K+kk[None,:]
        else:hp=(tg*HV+ih)*K*V+kk[None,:]*V+vv[:,None]
        hm=(vv[:,None]<V)&(kk[None,:]<K)
        hs=tl.load(H0+hp,hm,0).to(key.dtype)
        dq=tl.dot(dout,hs,allow_tf32=False)*(eg*SCALE)[:,None]
        dq+=tl.dot(ds,key,allow_tf32=False)
        tl.store(DQ+off[:,None]*K+kk[None,:],dq,km)
        if GATED:dg+=tl.sum(dq*query,1)
        de=-tl.dot(pp,hs,allow_tf32=False)*eg[:,None]
        left=tl.dot(dm,key,allow_tf32=False)
        erase=de+left
        db_piece=tl.sum(erase*key,1)
        db+=db_piece
        dk_res=erase*beta[:,None]
        if GATED:dg+=beta*db_piece
        right=tl.dot(tl.trans(dm),(key*beta[:,None]).to(key.dtype),allow_tf32=False)
        dk_res+=right
        if GATED:dg-=tl.sum(right*key,1)
        dhs=tl.load(DH+hp,hm,0).to(key.dtype)
        dk_state=tl.dot(r,dhs,allow_tf32=False)*er[:,None]
        if GATED:
            hf=tl.load(H0+hp,hm,0);dhf=tl.load(DH+hp,hm,0)
            dg_last+=tl.sum(hf*dhf)*en+tl.sum(dk_state*key)
        dk_direct=dk_state+tl.dot(tl.trans(ds),query,allow_tf32=False)
        if GATED:dg-=tl.sum(dk_direct*key,1)
        tl.store(DK+off[:,None]*K+kk[None,:],dk_direct+dk_res,km)
    tl.store(DB+off,db,valid)
    if GATED:
        dg=tl.where(tt==tl.minimum((it+1)*C,length)-1,dg+dg_last,dg)
        dg=tl.cumsum(tl.where(valid,dg,0),0,reverse=True)
        tl.store(DG+off,dg,valid)


def adjoint(q,k,v,g,beta,A,dr,r,h,dh,do,scale,state_v_first=False,
            cu_seqlens=None,chunk_indices=None):
    from fla.ops.utils import prepare_chunk_indices
    B,T,H,K=k.shape;HV=v.shape[2];V=v.shape[-1];C=A.shape[-1]
    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices=prepare_chunk_indices(cu_seqlens,C)
    nt=len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T,C)
    dq=torch.empty((B,T,HV,K),device=k.device,dtype=q.dtype)
    dk=torch.empty_like(dq);dv=torch.empty_like(v);db=torch.empty_like(beta)
    dg=torch.empty_like(g) if g is not None else None
    unified[(nt,B*HV)](q,k,v,g,beta,A,dr,r,h,dh,do,dq,dk,dv,db,dg,cu_seqlens,chunk_indices,
        T=T,H=H,HV=HV,K=K,V=V,C=C,NK=32,NV=triton.next_power_of_2(V),SCALE=scale,
        GATED=g is not None,VAR=cu_seqlens is not None,SVF=state_v_first,num_warps=4,num_stages=1)
    if H!=HV:
        dq=dq.reshape(B,T,H,HV//H,K).sum(3);dk=dk.reshape(B,T,H,HV//H,K).sum(3)
    return dq,dk,dv,dg,db
