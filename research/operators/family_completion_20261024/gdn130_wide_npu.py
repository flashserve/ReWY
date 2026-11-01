"""GDN paper adjoint with Vector-only endpoint with a uniform wider value contraction.

Fixed NK64/NV128 consumer and NV128 producer for every shape. FP32 scratch;
only public BF16 dot operands are rounded, same precision budget as residual.
"""
from common import torch,triton,launch
import triton.language as tl

@triton.jit
def producer_wide(A,DR,R,V0,BETA,DO,G,P,DV,DBV,DM,DS,CU,CI,
             T:tl.constexpr,B:tl.constexpr,HV:tl.constexpr,V:tl.constexpr,C:tl.constexpr,
             NV:tl.constexpr,VAR:tl.constexpr,TASKS:tl.constexpr,STEP:tl.constexpr):
    for task in range(tl.program_id(0),TASKS,STEP):
        it=task//(B*HV);bh=task%(B*HV);ih=bh%HV
        if VAR:
            seq=tl.load(CI+2*it);it=tl.load(CI+2*it+1);bos=tl.load(CU+seq);length=tl.load(CU+seq+1)-bos
        else:bos=(bh//HV)*T;length=T
        rr=tl.arange(0,C);tt=it*C+rr;off=(bos+tt)*HV+ih
        soff=(bh//HV)*HV*T+ih*T+tt if not VAR else ih*T+bos+tt
        beta=tl.load(BETA+soff,tt<length,0).to(tl.float32)
        dm=tl.zeros((C,C),tl.float32);ds=tl.zeros((C,C),tl.float32);db=tl.zeros((C,),tl.float32)
        for iv in range(tl.cdiv(V,NV)):
            vv=iv*NV+tl.arange(0,NV);vm=(tt[:,None]<length)&(vv[None,:]<V)
            at=tl.load(A+off[None,:]*C+rr[:,None],(tt[:,None]<length)&(tt[None,:]<length),0)
            dr=tl.load(DR+off[:,None]*V+vv[None,:],vm,0)
            p=tl.dot(at,dr,allow_tf32=False)
            value=tl.load(V0+off[:,None]*V+vv[None,:],vm,0)
            db+=tl.sum(p*value,1)
            tl.store(P+off[:,None]*V+vv[None,:],p,vm)
            tl.store(DV+off[:,None]*V+vv[None,:],p*beta[:,None],vm)
            rt=tl.load(R+off[None,:]*V+vv[:,None],(tt[None,:]<length)&(vv[:,None]<V),0)
            dm-=tl.dot(p.to(rt.dtype),rt,allow_tf32=False)
            dout=tl.load(DO+off[:,None]*V+vv[None,:],vm,0)
            rt2=tl.load(R+off[None,:]*V+vv[:,None],(tt[None,:]<length)&(vv[:,None]<V),0)
            ds+=tl.dot(dout,rt2,allow_tf32=False)
        gg=tl.load(G+soff,tt<length,0).to(tl.float32)
        pair=(tt[:,None]<length)&(tt[None,:]<length)
        fac=tl.exp2(tl.where(pair&(rr[:,None]>=rr[None,:]),gg[:,None]-gg[None,:],0))
        dm*=fac
        ds*=fac
        mp=tl.make_block_ptr(DM+task*C*C,(C,C),(C,1),(0,0),(C,C),(1,0))
        sp=tl.make_block_ptr(DS+task*C*C,(C,C),(C,1),(0,0),(C,C),(1,0))
        tl.store(mp,tl.where(pair&(rr[:,None]>rr[None,:]),dm,0))
        tl.store(sp,tl.where(pair&(rr[:,None]>=rr[None,:]),ds,0))
        tl.store(DBV+soff,db,tt<length)

@triton.jit
def consumer_wide(Q,K0,G,BETA,P,R,H0,DH,DO,DM,DS,DBV,DQ,DK,DB,DG,CU,CI,
             T:tl.constexpr,B:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
             C:tl.constexpr,NK:tl.constexpr,NV:tl.constexpr,SCALE:tl.constexpr,
             GATED:tl.constexpr,VAR:tl.constexpr,SVF:tl.constexpr,TASKS:tl.constexpr,STEP:tl.constexpr):
    for task in range(tl.program_id(0),TASKS,STEP):
        it=task//(B*HV);bh=task%(B*HV);ih=bh%HV
        if VAR:
            tg=it;seq=tl.load(CI+2*it).to(tl.int32);it=tl.load(CI+2*it+1).to(tl.int32);bos=tl.load(CU+seq);length=(tl.load(CU+seq+1)-bos).to(tl.int32)
        else:bos=(bh//HV)*T;length=T;tg=(bh//HV)*tl.cdiv(T,C)+it
        rr=tl.arange(0,C);tt=it*C+rr;off=(bos+tt)*HV+ih
        soff=(bh//HV)*HV*T+ih*T+tt if not VAR else ih*T+bos+tt
        valid=tt<length;pair=valid[:,None]&valid[None,:]
        beta=tl.load(BETA+soff,valid,0).to(tl.float32)
        db=tl.load(DBV+soff,valid,0).to(tl.float32);dg=tl.zeros((C,),tl.float32);end=tl.full((),0,tl.float32)
        if GATED:
            gg=tl.load(G+soff,valid,0).to(tl.float32)
            sn=(bh//HV)*HV*T+ih*T+tl.minimum((it+1)*C,length)-1 if not VAR else ih*T+bos+tl.minimum((it+1)*C,length)-1
            gn=tl.load(G+sn).to(tl.float32);eg=tl.exp2(gg);en=tl.exp2(gn)
            er=tl.where(valid,tl.exp2(gn-gg),0)
        else:
            eg=tl.full((C,),1,tl.float32);er=tl.full((C,),1,tl.float32)
        for ik in range(tl.cdiv(K,NK)):
            dq0=tl.zeros((C,NK),tl.float32);dk0=tl.zeros((C,NK),tl.float32);de=tl.zeros((C,NK),tl.float32)
            for iv in range(tl.cdiv(V,NV)):
                if SVF:
                    hp=tl.make_block_ptr(H0+(tg*HV+ih)*K*V,(V,K),(K,1),(iv*NV,ik*NK),(NV,NK),(1,0))
                    dp=tl.make_block_ptr(DH+(tg*HV+ih)*K*V,(V,K),(K,1),(iv*NV,ik*NK),(NV,NK),(1,0))
                else:
                    hp=tl.make_block_ptr(H0+(tg*HV+ih)*K*V,(V,K),(1,V),(iv*NV,ik*NK),(NV,NK),(0,1))
                    dp=tl.make_block_ptr(DH+(tg*HV+ih)*K*V,(V,K),(1,V),(iv*NV,ik*NK),(NV,NK),(0,1))
                hs=tl.load(hp,boundary_check=(0,1),padding_option='zero')
                dout=tl.load(tl.make_block_ptr(DO+(bos*HV+ih)*V,(length,V),(HV*V,1),(it*C,iv*NV),(C,NV),(1,0)),boundary_check=(0,1),padding_option='zero')
                dq0+=tl.dot(dout,hs.to(dout.dtype),allow_tf32=False)
                rv=tl.load(tl.make_block_ptr(R+(bos*HV+ih)*V,(length,V),(HV*V,1),(it*C,iv*NV),(C,NV),(1,0)),boundary_check=(0,1),padding_option='zero')
                dhs2=tl.load(dp,boundary_check=(0,1),padding_option='zero')
                dk0+=tl.dot(rv,dhs2.to(rv.dtype),allow_tf32=False)
                pp=tl.load(tl.make_block_ptr(P+(bos*HV+ih)*V,(length,V),(HV*V,1),(it*C,iv*NV),(C,NV),(1,0)),boundary_check=(0,1),padding_option='zero').to(K0.dtype.element_ty)
                hs2=tl.load(hp,boundary_check=(0,1),padding_option='zero').to(K0.dtype.element_ty)
                de-=tl.dot(pp,hs2,allow_tf32=False)
            kp=tl.make_block_ptr(K0+(bos*H+ih//(HV//H))*K,(length,K),(H*K,1),(it*C,ik*NK),(C,NK),(1,0))
            qp=tl.make_block_ptr(Q+(bos*H+ih//(HV//H))*K,(length,K),(H*K,1),(it*C,ik*NK),(C,NK),(1,0))
            key=tl.load(kp,boundary_check=(0,1),padding_option='zero')
            dk0*=er[:,None]
            if GATED:end+=tl.sum(dk0*key)
            ds=tl.load(tl.make_block_ptr(DS+task*C*C,(C,C),(C,1),(0,0),(C,C),(1,0)))
            ds=(ds*SCALE).to(key.dtype)
            dq=dq0*(eg*SCALE)[:,None]+tl.dot(ds,key,allow_tf32=False)
            dst=tl.load(tl.make_block_ptr(DS+task*C*C,(C,C),(1,C),(0,0),(C,C),(0,1)))
            query=tl.load(qp,boundary_check=(0,1),padding_option='zero')
            dk_direct=dk0+tl.dot((dst*SCALE).to(query.dtype),query,allow_tf32=False)
            dm=tl.load(tl.make_block_ptr(DM+task*C*C,(C,C),(C,1),(0,0),(C,C),(1,0)))
            key2=tl.load(kp,boundary_check=(0,1),padding_option='zero')
            left=tl.dot(dm,key2.to(tl.float32),allow_tf32=False)
            dmt=tl.load(tl.make_block_ptr(DM+task*C*C,(C,C),(1,C),(0,0),(C,C),(0,1)))
            key3=tl.load(kp,boundary_check=(0,1),padding_option='zero')
            right=tl.dot(dmt,key3.to(tl.float32)*beta[:,None],allow_tf32=False)
            key4=tl.load(kp,boundary_check=(0,1),padding_option='zero').to(tl.float32)
            db_piece=tl.sum((de*eg[:,None]+left)*key4,1)
            db+=db_piece
            dk=dk_direct+(de*eg[:,None]+left)*beta[:,None]+right
            if GATED:
                query2=tl.load(qp,boundary_check=(0,1),padding_option='zero').to(tl.float32)
                # Homogeneity identity reuses the already reduced beta term.
                # It also shortens the matrix live ranges below the UB limit.
                dg+=tl.sum(dq*query2-dk*key4,1)+2*beta*db_piece
            tl.store(tl.make_block_ptr(DQ+(bos*HV+ih)*K,(length,K),(HV*K,1),(it*C,ik*NK),(C,NK),(1,0)),dq.to(DQ.dtype.element_ty),boundary_check=(0,1))
            tl.store(tl.make_block_ptr(DK+(bos*HV+ih)*K,(length,K),(HV*K,1),(it*C,ik*NK),(C,NK),(1,0)),dk.to(DK.dtype.element_ty),boundary_check=(0,1))
        tl.store(DB+soff,db,valid)
        if GATED:
            dg=tl.where(tt==tl.minimum((it+1)*C,length)-1,dg+end,dg)
            tl.store(DG+soff,dg,valid)

@triton.jit
def finalize_gate_wide(H0,DH,G,PART,DG,CU,CI,
                  T:tl.constexpr,B:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
                  C:tl.constexpr,BX:tl.constexpr,VAR:tl.constexpr,TASKS:tl.constexpr,STEP:tl.constexpr):
    for task in range(tl.program_id(0),TASKS,STEP):
        it=task//(B*HV);bh=task%(B*HV);ih=bh%HV
        if VAR:
            tg=it
            seq=tl.load(CI+2*it).to(tl.int32)
            it=tl.load(CI+2*it+1).to(tl.int32)
            bos=tl.load(CU+seq)
            length=(tl.load(CU+seq+1)-bos).to(tl.int32)
        else:
            bos=(bh//HV)*T;length=T;tg=(bh//HV)*tl.cdiv(T,C)+it
        # Frobenius product is invariant to K/V storage order. Read both flat.
        base=(tg*HV+ih)*K*V
        xx=tl.arange(0,BX)
        acc=tl.zeros((BX,),tl.float32)
        for ix in range(tl.cdiv(K*V,BX)):
            x=ix*BX+xx
            hh=tl.load(H0+base+x,x<K*V,0).to(tl.float32)
            jj=tl.load(DH+base+x,x<K*V,0).to(tl.float32)
            acc+=hh*jj
        end=tl.sum(acc,0)
        rr=tl.arange(0,C);tt=it*C+rr;valid=tt<length
        soff=(bh//HV)*HV*T+ih*T+tt if not VAR else ih*T+bos+tt
        sn=(bh//HV)*HV*T+ih*T+tl.minimum((it+1)*C,length)-1 if not VAR else ih*T+bos+tl.minimum((it+1)*C,length)-1
        gn=tl.load(G+sn).to(tl.float32)
        part=tl.load(PART+soff,valid,0).to(tl.float32)
        part=tl.where(tt==tl.minimum((it+1)*C,length)-1,part+end*tl.exp2(gn),part)
        dg=tl.cumsum(tl.where(valid,part,0),0,reverse=True)
        tl.store(DG+soff,dg,valid)

def adjoint(q,k,v,g,beta,A,dr,r,h,dh,do,scale,state_v_first=False,cu_seqlens=None,chunk_indices=None):
    assert g is not None
    from fla.ops.utils import prepare_chunk_indices
    B,T,H,K=k.shape;HV=v.shape[2];V=v.shape[-1];C=A.shape[-1]
    if cu_seqlens is not None and chunk_indices is None:chunk_indices=prepare_chunk_indices(cu_seqlens,C)
    nt=len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T,C)
    p=torch.empty_like(v,dtype=torch.float32);dm=torch.empty((nt*B*HV,C,C),device=q.device,dtype=torch.float32);ds=torch.empty_like(dm)
    dv=torch.empty_like(v);dq=torch.empty((B,T,HV,K),device=q.device,dtype=q.dtype);dk=torch.empty_like(dq)
    bc=beta.transpose(1,2).contiguous();gc=g.transpose(1,2).contiguous() if g is not None else None
    dbv=torch.empty_like(bc,dtype=torch.float32);db=torch.empty_like(bc);dg=torch.empty_like(gc) if g is not None else None
    dg_partial=torch.empty_like(gc)
    shared=dict(T=T,B=B,HV=HV,V=V,C=C,VAR=cu_seqlens is not None,CU=cu_seqlens,CI=chunk_indices)
    launch(producer_wide,nt*B*HV,A=A,DR=dr,R=r,V0=v,BETA=bc,DO=do,G=gc,P=p,DV=dv,DBV=dbv,DM=dm,DS=ds,NV=128,**shared)
    launch(consumer_wide,nt*B*HV,Q=q,K0=k,G=gc,BETA=bc,P=p,R=r,H0=h,DH=dh,DO=do,DM=dm,DS=ds,DBV=dbv,DQ=dq,DK=dk,DB=db,DG=dg_partial,
        H=H,K=K,NK=64,NV=128,SCALE=scale,GATED=g is not None,SVF=state_v_first,**shared)
    from fla.utils import ascend_compile_kwargs
    from fla.utils.ascend_ub_manager import get_npu_properties
    nvec=get_npu_properties()['num_vectorcore']
    finalize_gate_wide[(nvec,)](H0=h,DH=dh,G=gc,PART=dg_partial,DG=dg,K=K,BX=2048,
        TASKS=nt*B*HV,STEP=nvec,**shared,**ascend_compile_kwargs())
    if H!=HV:dq=dq.reshape(B,T,H,HV//H,K).sum(3);dk=dk.reshape(B,T,H,HV//H,K).sum(3)
    return dq,dk,dv,dg.transpose(1,2).contiguous() if dg is not None else None,db.transpose(1,2).contiguous()
