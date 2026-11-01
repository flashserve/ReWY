"""Residual WY for scalar-beta DeltaNet/GDN; explicit bounded tile lifetimes.

Producer owns P, dM, direct value/beta outputs. Per-key-tile consumers own
disjoint dk and partial scalar reductions. Final reducer merges the partials.
NPU uses persistent physical-core tasks; GPU uses independent tile programs.
No division by beta and no change to the forward state scan.
"""
from common import torch, triton, launch, DEV
import triton.language as tl

@triton.jit
def producer(A, DR, R, V0, BETA, P, DV, DBV, DM, CU, CI,
             T:tl.constexpr,B:tl.constexpr,HV:tl.constexpr,V:tl.constexpr,
             C:tl.constexpr,NV:tl.constexpr,VAR:tl.constexpr,TASKS:tl.constexpr,STEP:tl.constexpr,VECTOR:tl.constexpr=False):
    for task in range(tl.program_id(0),TASKS,STEP):
        it=task//(B*HV); bh=task%(B*HV); ih=bh%HV
        if VAR:
            seq=tl.load(CI+2*it); it=tl.load(CI+2*it+1)
            bos=tl.load(CU+seq); length=tl.load(CU+seq+1)-bos
        else:
            bos=(bh//HV)*T; length=T
        rr=tl.arange(0,C); tt=it*C+rr; off=(bos+tt)*HV+ih
        dd=tl.zeros((C,C),tl.float32); db=tl.zeros((C,),tl.float32)
        if not VECTOR:beta=tl.load(BETA+off,tt<length,other=0).to(tl.float32)
        for iv in range(tl.cdiv(V,NV)):
            vv=iv*NV+tl.arange(0,NV)
            at=tl.load(A+off[None,:]*C+rr[:,None],(tt[None,:]<length)&(tt[:,None]<length),other=0)
            dr=tl.load(DR+off[:,None]*V+vv[None,:],(tt[:,None]<length)&(vv[None,:]<V),other=0)
            p=tl.dot(at,dr,allow_tf32=False)
            value=tl.load(V0+off[:,None]*V+vv[None,:],(tt[:,None]<length)&(vv[None,:]<V),other=0)
            if VECTOR:
                beta=tl.load(BETA+off[:,None]*V+vv[None,:],(tt[:,None]<length)&(vv[None,:]<V),other=0).to(tl.float32)
                tl.store(DBV+off[:,None]*V+vv[None,:],p*value,(tt[:,None]<length)&(vv[None,:]<V))
            else:db+=tl.sum(p*value,1)
            tl.store(P+off[:,None]*V+vv[None,:],p,(tt[:,None]<length)&(vv[None,:]<V))
            if VECTOR:out=p*beta
            else:out=p*beta[:,None]
            tl.store(DV+off[:,None]*V+vv[None,:],out,(tt[:,None]<length)&(vv[None,:]<V))
            rt=tl.load(R+off[None,:]*V+vv[:,None],(tt[None,:]<length)&(vv[:,None]<V),other=0)
            dd-=tl.dot(p.to(rt.dtype),rt,allow_tf32=False)
        dd=tl.where((rr[:,None]>rr[None,:])&(tt[:,None]<length)&(tt[None,:]<length),dd,0)
        tl.store(DM+off[:,None]*C+rr[None,:],dd,tt[:,None]<length)
        if not VECTOR:tl.store(DBV+off,db,tt<length)

@triton.jit
def consumer(K0,G,BETA,P,DM,H0,DK,DBP,DGP,CU,CI,
             T:tl.constexpr,B:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
             C:tl.constexpr,NK:tl.constexpr,NV:tl.constexpr,VAR:tl.constexpr,
             GATED:tl.constexpr,GRAM:tl.constexpr,SVF:tl.constexpr,TASKS:tl.constexpr,STEP:tl.constexpr):
    for task in range(tl.program_id(0),TASKS,STEP):
        ik=task%tl.cdiv(K,NK); rest=task//tl.cdiv(K,NK)
        it=rest//(B*HV); bh=rest%(B*HV); ih=bh%HV
        if VAR:
            tg=it;seq=tl.load(CI+2*it);it=tl.load(CI+2*it+1)
            bos=tl.load(CU+seq);length=tl.load(CU+seq+1)-bos
        else:
            bos=(bh//HV)*T;length=T;tg=(bh//HV)*tl.cdiv(T,C)+it
        rr=tl.arange(0,C);tt=it*C+rr;kk=ik*NK+tl.arange(0,NK)
        off=(bos+tt)*HV+ih;ko=(bos+tt)*H+ih//(HV//H)
        beta=tl.load(BETA+off,tt<length,other=0).to(tl.float32)
        if GATED:
            gg=tl.load(G+off,tt<length,other=0).to(tl.float32)
            eg=tl.exp2(gg)
        else:
            gg=tl.full((C,),0,tl.float32);eg=tl.full((C,),1,tl.float32)
        de=tl.zeros((C,NK),tl.float32)
        for iv in range(tl.cdiv(V,NV)):
            vv=iv*NV+tl.arange(0,NV)
            pp=tl.load(P+off[:,None]*V+vv[None,:],(tt[:,None]<length)&(vv[None,:]<V),other=0).to(K0.dtype.element_ty)
            if SVF:
                hp=(tg*HV+ih)*K*V+vv[:,None]*K+kk[None,:]
            else:
                hp=(tg*HV+ih)*K*V+kk[None,:]*V+vv[:,None]
            hh=tl.load(H0+hp,(vv[:,None]<V)&(kk[None,:]<K),other=0).to(K0.dtype.element_ty)
            de-=tl.dot(pp,hh,allow_tf32=False)
        dd=tl.load(DM+off[:,None]*C+rr[None,:],(tt[:,None]<length)&(tt[None,:]<length),other=0)
        dd=tl.where(rr[:,None]>rr[None,:],dd*tl.exp2(gg[:,None]-gg[None,:]),0).to(K0.dtype.element_ty)
        key=tl.load(K0+ko[:,None]*K+kk[None,:],(tt[:,None]<length)&(kk[None,:]<K),other=0)
        left=tl.dot(dd,key,allow_tf32=False)
        # Independent transposed loads avoid the unstable shared-operand path on Ascend.
        dt=tl.load(DM+off[None,:]*C+rr[:,None],(tt[None,:]<length)&(tt[:,None]<length),other=0)
        dt=tl.where(rr[None,:]>rr[:,None],dt*tl.exp2(gg[None,:]-gg[:,None]),0).to(K0.dtype.element_ty)
        key2=tl.load(K0+ko[:,None]*K+kk[None,:],(tt[:,None]<length)&(kk[None,:]<K),other=0)
        right=tl.dot(dt,(key2*beta[:,None]).to(key2.dtype),allow_tf32=False)
        key3=tl.load(K0+ko[:,None]*K+kk[None,:],(tt[:,None]<length)&(kk[None,:]<K),other=0).to(tl.float32)
        dk=de*(beta*eg)[:,None]+beta[:,None]*left+right
        db=tl.sum(de*key3*eg[:,None]+left*key3,1)
        dg=tl.sum(de*key3*(beta*eg)[:,None],1)
        if GATED:
            if GRAM:
                ka=tl.load(K0+ko[:,None]*K+kk[None,:],(tt[:,None]<length)&(kk[None,:]<K),other=0)
                kb=tl.load(K0+ko[None,:]*K+kk[:,None],(tt[None,:]<length)&(kk[:,None]<K),other=0)
                gram=tl.dot(ka,kb,allow_tf32=False)
                ds=tl.load(DM+off[:,None]*C+rr[None,:],(tt[:,None]<length)&(tt[None,:]<length),other=0)
                ds=tl.where(rr[:,None]>rr[None,:],ds*tl.exp2(gg[:,None]-gg[None,:]),0).to(K0.dtype.element_ty).to(tl.float32)
                j=gram*ds*beta[:,None]
                dg+=tl.sum(j,1)-tl.sum(j,0)
            else:
                dg+=tl.sum(key3*(beta[:,None]*left-right),1)
        tl.store(DK+off[:,None]*K+kk[None,:],dk,(tt[:,None]<length)&(kk[None,:]<K))
        tl.store(DBP+off*tl.cdiv(K,NK)+ik,db,tt<length)
        if GATED:tl.store(DGP+off*tl.cdiv(K,NK)+ik,dg,tt<length)

@triton.jit
def reduce_scalar(DBV,DBP,DGP,DB,DG,N:tl.constexpr,PART:tl.constexpr,NP:tl.constexpr,GATED:tl.constexpr,
                  BLOCK:tl.constexpr,TASKS:tl.constexpr,STEP:tl.constexpr):
    for task in range(tl.program_id(0),TASKS,STEP):
        i=task*BLOCK+tl.arange(0,BLOCK);j=tl.arange(0,NP)
        b=tl.load(DBP+i[:,None]*PART+j[None,:],(i[:,None]<N)&(j[None,:]<PART),other=0)
        base=tl.load(DBV+i,i<N,other=0)
        tl.store(DB+i,base+tl.sum(b,1),i<N)
        if GATED:
            g=tl.load(DGP+i[:,None]*PART+j[None,:],(i[:,None]<N)&(j[None,:]<PART),other=0)
            tl.store(DG+i,tl.sum(g,1),i<N)

def wy(k,v,beta,A,du,h,r,g=None,gram=False,state_v_first=False,cu_seqlens=None,chunk_indices=None,**unused):
    B,T,H,K=k.shape;HV=v.shape[2];V=v.shape[-1];C=A.shape[-1]
    from fla.ops.utils import prepare_chunk_indices
    if cu_seqlens is not None and chunk_indices is None:chunk_indices=prepare_chunk_indices(cu_seqlens,C)
    nt=len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T,C)
    nk=32 if DEV=='npu' else 64;nv=32 if DEV=='npu' else 64
    p=torch.empty_like(v,dtype=torch.float32);dm=torch.empty_like(A,dtype=torch.float32)
    dv=torch.empty_like(v);dbv=torch.empty_like(beta,dtype=torch.float32)
    dk=torch.empty((B,T,HV,K),device=k.device,dtype=k.dtype)
    part=triton.cdiv(K,nk)
    dbp=torch.empty((B,T,HV,part),device=k.device,dtype=torch.float32);dgp=torch.empty_like(dbp)
    db=torch.empty_like(beta);dg=torch.empty_like(g) if g is not None else None
    common=dict(T=T,B=B,HV=HV,V=V,C=C,VAR=cu_seqlens is not None,CU=cu_seqlens,CI=chunk_indices)
    launch(producer,nt*B*HV,A=A,DR=du,R=r,V0=v,BETA=beta,P=p,DV=dv,DBV=dbv,DM=dm,NV=nv,**common)
    if DEV=='npu' and g is not None:
        from scalar_rows_npu import consumer as row_consumer
        launch(row_consumer,nt*B*HV*part*triton.cdiv(C,16),K0=k,G=g,BETA=beta,P=p,DM=dm,H0=h,DK=dk,DBP=dbp,DGP=dgp,
               H=H,K=K,NK=nk,NV=nv,NR=16,GRAM=gram,SVF=state_v_first,**common)
    else:
        launch(consumer,nt*B*HV*part,K0=k,G=g,BETA=beta,P=p,DM=dm,H0=h,DK=dk,DBP=dbp,DGP=dgp,
               H=H,K=K,NK=nk,NV=nv,GATED=g is not None,GRAM=gram,SVF=state_v_first,**common)
    launch(reduce_scalar,triton.cdiv(B*T*HV,128),DBV=dbv,DBP=dbp,DGP=dgp,DB=db,DG=dg,
           N=B*T*HV,PART=part,NP=triton.next_power_of_2(part),GATED=g is not None,BLOCK=128)
    if H!=HV:dk=dk.reshape(B,T,H,HV//H,K).sum(3)
    return (dk,dv,db,dg) if g is not None else (dk,dv,db)
