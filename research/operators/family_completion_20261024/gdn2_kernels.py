"""Vector erase/write residual WY and independently scheduled matrix intra."""
from common import *
import triton.language as tl
from residual_scalar import producer
from gdn2_fallback_npu import body as subchunk_fallback

@triton.jit
def state_vjp(Q,K0,G,B0,P,R,H0,DH,DO,DQ,DK,DB,DG,CU,CI,
              T:tl.constexpr,B:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,C:tl.constexpr,
              NK:tl.constexpr,NV:tl.constexpr,VAR:tl.constexpr,SVF:tl.constexpr,SCALE:tl.constexpr,
              TASKS:tl.constexpr,STEP:tl.constexpr):
    for task in range(tl.program_id(0),TASKS,STEP):
        ik=task%tl.cdiv(K,NK);rest=task//tl.cdiv(K,NK);it=rest//(B*H);bh=rest%(B*H);ih=bh%H
        if VAR:
            tg=it;seq=tl.load(CI+2*it);it=tl.load(CI+2*it+1);bos=tl.load(CU+seq);length=tl.load(CU+seq+1)-bos
        else:bos=(bh//H)*T;length=T;tg=(bh//H)*tl.cdiv(T,C)+it
        rr=tl.arange(0,C);tt=it*C+rr;kk=ik*NK+tl.arange(0,NK)
        off=(bos+tt)*H+ih;mask=(tt[:,None]<length)&(kk[None,:]<K)
        de=tl.zeros((C,NK),tl.float32);dq=tl.zeros((C,NK),tl.float32);dk=tl.zeros((C,NK),tl.float32);end=tl.zeros((NK,),tl.float32)
        for iv in range(tl.cdiv(V,NV)):
            vv=iv*NV+tl.arange(0,NV);vm=(tt[:,None]<length)&(vv[None,:]<V);hm=(vv[:,None]<V)&(kk[None,:]<K)
            if SVF:hp=(tg*H+ih)*K*V+vv[:,None]*K+kk[None,:]
            else:hp=(tg*H+ih)*K*V+kk[None,:]*V+vv[:,None]
            hs=tl.load(H0+hp,hm,0);ds=tl.load(DH+hp,hm,0)
            end+=tl.sum(hs*ds,0)
            ov=tl.load(DO+off[:,None]*V+vv[None,:],vm,0)
            dq+=tl.dot(ov,hs.to(ov.dtype),allow_tf32=False)
            rv=tl.load(R+off[:,None]*V+vv[None,:],vm,0)
            dk+=tl.dot(rv,ds.to(rv.dtype),allow_tf32=False)
            pp=tl.load(P+off[:,None]*V+vv[None,:],vm,0).to(K0.dtype.element_ty)
            hs2=tl.load(H0+hp,hm,0).to(K0.dtype.element_ty)
            de-=tl.dot(pp,hs2,allow_tf32=False)
        gg=tl.load(G+off[:,None]*K+kk[None,:],mask,0).to(tl.float32)
        ge=tl.load(G+((bos+tl.minimum(length,it*C+C)-1)*H+ih)*K+kk,kk<K,0).to(tl.float32)
        q=tl.load(Q+off[:,None]*K+kk[None,:],mask,0).to(tl.float32)
        k=tl.load(K0+off[:,None]*K+kk[None,:],mask,0).to(tl.float32)
        b=tl.load(B0+off[:,None]*K+kk[None,:],mask,0).to(tl.float32)
        eg=tl.exp2(gg);dq*=eg*SCALE;dk*=tl.where(mask,tl.exp2(ge[None,:]-gg),0)
        kd=k*dk;end=end*tl.exp2(ge)+tl.sum(kd,0)
        dg=q*dq-kd+(tt==tl.minimum(length,it*C+C)-1)[:,None]*end[None,:]+k*eg*de*b
        tl.store(DQ+off[:,None]*K+kk[None,:],dq,mask)
        tl.store(DK+off[:,None]*K+kk[None,:],dk+de*eg*b,mask)
        tl.store(DB+off[:,None]*K+kk[None,:],de*k*eg,mask)
        tl.store(DG+off[:,None]*K+kk[None,:],dg,mask)

@triton.jit
def matrix(Q,K0,G,B0,L,D,DQ,DK,DB,DG,CU,CI,
           T:tl.constexpr,B:tl.constexpr,H:tl.constexpr,K:tl.constexpr,C:tl.constexpr,NK:tl.constexpr,
           VAR:tl.constexpr,MAX_SPAN:tl.constexpr,NPU_FALLBACK:tl.constexpr,TASKS:tl.constexpr,STEP:tl.constexpr):
    for task in range(tl.program_id(0),TASKS,STEP):
        ik=task%tl.cdiv(K,NK);rest=task//tl.cdiv(K,NK);it=rest//(B*H);bh=rest%(B*H);ih=bh%H
        global_chunk=it
        if VAR:
            seq=tl.load(CI+2*it);it=tl.load(CI+2*it+1);bos=tl.load(CU+seq);length=tl.load(CU+seq+1)-bos
        else:bos=(bh//H)*T;length=T
        rr=tl.arange(0,C);tt=it*C+rr;kk=ik*NK+tl.arange(0,NK);off=(bos+tt)*H+ih
        mask=(tt[:,None]<length)&(kk[None,:]<K);pair=(tt[:,None]<length)&(tt[None,:]<length)
        gg=tl.load(G+off[:,None]*K+kk[None,:],mask,0).to(tl.float32)
        lo=tl.min(tl.where(tt[:,None]<length,gg,float('inf')),0);hi=tl.max(tl.where(tt[:,None]<length,gg,-float('inf')),0)
        matrix_ok=tl.max(hi-lo,0)<MAX_SPAN
        if matrix_ok:
            center=(lo+hi)*.5;plus=tl.where(mask,tl.exp2(gg-center[None,:]),0);minus=tl.where(mask,tl.exp2(center[None,:]-gg),0)
            k=tl.load(K0+off[:,None]*K+kk[None,:],mask,0).to(tl.float32)
            l=tl.load(L+off[:,None]*C+rr[None,:],pair&(rr[:,None]>=rr[None,:]),0).to(tl.bfloat16)
            uq=tl.dot(l,(k*minus).to(tl.bfloat16),allow_tf32=False)*plus
            d=tl.load(D+off[:,None]*C+rr[None,:],pair&(rr[:,None]>rr[None,:]),0).to(tl.bfloat16)
            k2=tl.load(K0+off[:,None]*K+kk[None,:],mask,0).to(tl.float32)
            ue=tl.dot(d,(k2*minus).to(tl.bfloat16),allow_tf32=False)*plus
            lt=tl.load(L+off[None,:]*C+rr[:,None],pair&(rr[None,:]>=rr[:,None]),0).to(tl.bfloat16)
            q2=tl.load(Q+off[:,None]*K+kk[None,:],mask,0).to(tl.float32)
            uk=tl.dot(lt,(q2*plus).to(tl.bfloat16),allow_tf32=False)
            dt=tl.load(D+off[None,:]*C+rr[:,None],pair&(rr[None,:]>rr[:,None]),0).to(tl.bfloat16)
            k3=tl.load(K0+off[:,None]*K+kk[None,:],mask,0).to(tl.float32)
            b3=tl.load(B0+off[:,None]*K+kk[None,:],mask,0).to(tl.float32)
            uk=(uk+tl.dot(dt,(k3*b3*plus).to(tl.bfloat16),allow_tf32=False))*minus
        elif NPU_FALLBACK:
            for piece in range(tl.cdiv(C,16)):
                subchunk_fallback(ik*tl.cdiv(C,16)+piece,global_chunk,bh,
                    q=Q,k=K0,g=G,b=B0,dAqk=L,dAkk=D,dq=DQ,dq2=DQ,dk=DK,dk2=DK,dg=DG,dg2=DG,db=DB,
                    cu_seqlens=CU,chunk_indices=CI,B=B,T=T,H=H,K=K,BT=C,BC=16,BK=NK,NC=tl.cdiv(C,16),
                    IS_VARLEN=VAR,SAFE_GATE=False,GATE_SHIFT=0.,USE_GATHER=False)
                tl.debug_barrier()
            uq=tl.zeros((C,NK),tl.float32);ue=tl.zeros((C,NK),tl.float32);uk=tl.zeros((C,NK),tl.float32)
        else:
            uq=tl.zeros((C,NK),tl.float32);ue=tl.zeros((C,NK),tl.float32);uk=tl.zeros((C,NK),tl.float32)
            for j in range(tl.minimum(C,length-it*C)):
                src=(bos+it*C+j)*H+ih
                kj=tl.load(K0+src*K+kk,kk<K,0).to(tl.float32);qj=tl.load(Q+src*K+kk,kk<K,0).to(tl.float32)
                gj=tl.load(G+src*K+kk,kk<K,0).to(tl.float32);bj=tl.load(B0+src*K+kk,kk<K,0).to(tl.float32)
                lc=tl.load(L+off*C+j,(tt<length)&(rr>=j),0);dc=tl.load(D+off*C+j,(tt<length)&(rr>j),0)
                lr=tl.load(L+src*C+rr,(tt<length)&(rr<=j),0);dr=tl.load(D+src*C+rr,(tt<length)&(rr<j),0)
                past=tl.exp2(tl.where(mask&(rr[:,None]>=j),gg-gj[None,:],0));future=tl.exp2(tl.where(mask&(rr[:,None]<=j),gj[None,:]-gg,0))
                uq+=lc[:,None]*kj[None,:]*past;ue+=dc[:,None]*kj[None,:]*past
                uk+=(lr[:,None]*qj[None,:]+dr[:,None]*kj[None,:]*bj[None,:])*future
        if matrix_ok or not NPU_FALLBACK:
            q=tl.load(Q+off[:,None]*K+kk[None,:],mask,0).to(tl.float32);k=tl.load(K0+off[:,None]*K+kk[None,:],mask,0).to(tl.float32)
            b=tl.load(B0+off[:,None]*K+kk[None,:],mask,0).to(tl.float32)
            oq=tl.load(DQ+off[:,None]*K+kk[None,:],mask,0);ok=tl.load(DK+off[:,None]*K+kk[None,:],mask,0)
            ob=tl.load(DB+off[:,None]*K+kk[None,:],mask,0);og=tl.load(DG+off[:,None]*K+kk[None,:],mask,0)
            tl.store(DQ+off[:,None]*K+kk[None,:],oq+uq,mask);tl.store(DK+off[:,None]*K+kk[None,:],ok+b*ue+uk,mask)
            tl.store(DB+off[:,None]*K+kk[None,:],ob+ue*k,mask);tl.store(DG+off[:,None]*K+kk[None,:],og+q*uq+k*(b*ue-uk),mask)

def residual(q,k,v,v_new,g,b,w_gate,A,h,do,dh,dv,scale=None,cu_seqlens=None,chunk_size=64,chunk_indices=None,state_v_first=False):
    B,T,H,K=k.shape;V=v.shape[-1];C=chunk_size
    from fla.ops.utils import prepare_chunk_indices
    if cu_seqlens is not None and chunk_indices is None:chunk_indices=prepare_chunk_indices(cu_seqlens,C)
    nt=len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T,C)
    nk=32;nv=32 if DEV=='npu' else 64
    p=torch.empty_like(v,dtype=torch.float32);dm=torch.empty_like(A,dtype=torch.float32);dv2=torch.empty_like(v);dw=torch.empty_like(w_gate,dtype=torch.float32)
    dq=torch.empty_like(q,dtype=torch.float32);dk=torch.empty_like(k,dtype=torch.float32);db=torch.empty_like(b,dtype=torch.float32);dg=torch.empty_like(g,dtype=torch.float32)
    common=dict(T=T,B=B,V=V,C=C,VAR=cu_seqlens is not None,CU=cu_seqlens,CI=chunk_indices)
    launch(producer,nt*B*H,A=A,DR=dv,R=v_new,V0=v,BETA=w_gate,P=p,DV=dv2,DBV=dw,DM=dm,NV=nv,HV=H,VECTOR=True,**common)
    launch(state_vjp,nt*B*H*triton.cdiv(K,nk),Q=q,K0=k,G=g,B0=b,P=p,R=v_new,H0=h,DH=dh,DO=do,DQ=dq,DK=dk,DB=db,DG=dg,
           H=H,K=K,NK=nk,NV=nv,SVF=state_v_first,SCALE=scale,**common)
    return dq,dk,dv2,db,dw,dg,dm

def matrix_intra(q,k,g,b,dAqk,dAkk,dq,dk,db,dg,cu_seqlens=None,chunk_indices=None,chunk_size=64,**unused):
    B,T,H,K=k.shape;C=chunk_size;nk=32
    from fla.ops.utils import prepare_chunk_indices
    if cu_seqlens is not None and chunk_indices is None:chunk_indices=prepare_chunk_indices(cu_seqlens,C)
    nt=len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T,C)
    launch(matrix,nt*B*H*triton.cdiv(K,nk),Q=q,K0=k,G=g,B0=b,L=dAqk,D=dAkk,DQ=dq,DK=dk,DB=db,DG=dg,
           CU=cu_seqlens,CI=chunk_indices,T=T,B=B,H=H,K=K,C=C,NK=nk,VAR=cu_seqlens is not None,
           MAX_SPAN=80.,NPU_FALLBACK=DEV=='npu')
    return dq,dk,db,dg
