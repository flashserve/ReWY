"""Whole-chunk intra consumers with bounded exponential factoring and fallback."""
import os
import torch
import triton
import triton.language as tl
from fla.ops.utils import prepare_chunk_indices
from fla.ops.utils.op import exp2
from fla.utils import IS_NPU, ascend_compile_kwargs
from kernels.kda_joint_helpers import residual_dt_tile


@triton.jit(do_not_specialize=['T'])
def fused_kernel(q,k,g,beta,dAqk,P,R,dq,dk,dg,db,cu,ci,T,
                HSTATE,DHSTATE,DOUT,PP,CO,scale,HT,
                H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
                VP:tl.constexpr,BT:tl.constexpr,BK:tl.constexpr,VARLEN:tl.constexpr,
                MATERIALIZED:tl.constexpr,BETA_COMPLETE:tl.constexpr,
                REVERSE_CUMSUM:tl.constexpr,DOT_PRECISION:tl.constexpr,LOWP:tl.constexpr,VF:tl.constexpr,
                COMPENSATE:tl.constexpr,LOGIT_SCALE:tl.constexpr,ENDPOINT:tl.constexpr):
    nk=tl.cdiv(K,BK)
    it=tl.program_id(0)//nk
    ik=tl.program_id(0)%nk
    ibh=tl.program_id(1)
    ih=ibh%HV
    ikh=ih//(HV//H)
    if VARLEN:
        seq=tl.load(ci+2*it).to(tl.int32)
        it=tl.load(ci+2*it+1).to(tl.int32)
        bos=tl.load(cu+seq).to(tl.int64)
        length=tl.load(cu+seq+1).to(tl.int64)-bos
    else:
        bos=(ibh//HV).to(tl.int64)*T
        length=T
    idx=tl.arange(0,BT)
    tt=it*BT+idx
    kk=ik*BK+tl.arange(0,BK)
    valid=(tt[:,None]<length)&(kk[None,:]<K)
    qs=((bos+tt[:,None])*H+ikh)*K+kk[None,:]
    gs=((bos+tt[:,None])*HV+ih)*K+kk[None,:]
    qv=tl.load(q+qs,mask=valid,other=0.).to(tl.float32)
    kv=tl.load(k+qs,mask=valid,other=0.).to(tl.float32)
    gv=tl.load(g+gs,mask=valid,other=0.).to(tl.float32)
    bv=tl.load(beta+(bos+tt)*HV+ih,mask=tt<length,other=0.).to(tl.float32)
    off=((bos+tt[:,None])*HV+ih)*BT+idx[None,:]
    aq=tl.load(dAqk+off,mask=tt[:,None]<length,other=0.).to(tl.float32)
    aq=tl.where((idx[:,None]>=idx[None,:])&(tt[None,:]<length),aq,0.)
    if MATERIALIZED:
        dt=tl.load(P+off,mask=tt[:,None]<length,other=0.).to(tl.float32)
    else:
        dt=residual_dt_tile(P,R,bos,ih,tt,tt,length,HV,V,VP)
    gmin=tl.min(tl.where(tt[:,None]<length,gv,float('inf')),0)
    gmax=tl.max(tl.where(tt[:,None]<length,gv,-float('inf')),0)
    factor_ok=tl.max(gmax-gmin,0)<80.
    if factor_ok:
        anchor=(gmin+gmax)*0.5
        gp=tl.where(valid,exp2(gv-anchor[None,:]),0.)
        gm=tl.where(valid,exp2(anchor[None,:]-gv),0.)
        km=kv*gm
        qp=qv*gp
        kp=kv*bv[:,None]*gp
        if LOWP:
            aq_dot=aq.to(k.dtype.element_ty)
            dt_dot=dt.to(k.dtype.element_ty)
            km=km.to(k.dtype.element_ty)
            qp=qp.to(k.dtype.element_ty)
            kp=kp.to(k.dtype.element_ty)
        else:
            aq_dot=aq
            dt_dot=dt
        dq_add=tl.dot(aq_dot,km,input_precision=DOT_PRECISION)*gp
        de_add=tl.dot(dt_dot,km,input_precision=DOT_PRECISION)*gp
        dy_add=(tl.dot(tl.trans(aq_dot),qp,input_precision=DOT_PRECISION)+
                tl.dot(tl.trans(dt_dot),kp,input_precision=DOT_PRECISION))*gm
    else:
        dq_add=tl.zeros((BT,BK),tl.float32)
        de_add=tl.zeros((BT,BK),tl.float32)
        dy_add=tl.zeros((BT,BK),tl.float32)
        for j in range(0,min(BT,length-it*BT)):
            kval=tl.load(k+((bos+it*BT+j)*H+ikh)*K+kk,mask=kk<K,other=0.).to(tl.float32)
            qval=tl.load(q+((bos+it*BT+j)*H+ikh)*K+kk,mask=kk<K,other=0.).to(tl.float32)
            gval=tl.load(g+((bos+it*BT+j)*HV+ih)*K+kk,mask=kk<K,other=0.).to(tl.float32)
            bval=tl.load(beta+(bos+it*BT+j)*HV+ih).to(tl.float32)
            aqcol=tl.sum(tl.where(idx[None,:]==j,aq,0.),1)
            dtcol=tl.sum(tl.where(idx[None,:]==j,dt,0.),1)
            aqrow=tl.sum(tl.where(idx[:,None]==j,aq,0.),0)
            dtrow=tl.sum(tl.where(idx[:,None]==j,dt,0.),0)
            past=exp2(tl.where(idx[:,None]>=j,gv-gval[None,:],0.))
            future=exp2(tl.where(idx[:,None]<=j,gval[None,:]-gv,0.))
            dq_add+=aqcol[:,None]*kval[None,:]*past
            de_add+=dtcol[:,None]*kval[None,:]*past
            dy_add+=(aqrow[:,None]*qval[None,:]+dtrow[:,None]*kval[None,:]*bval)*future
    d_beta=tl.sum(de_add*kv,1)
    if LOGIT_SCALE>0: d_beta*=bv*(1.-bv/LOGIT_SCALE)
    if BETA_COMPLETE:
        beta_mask=(tt<length)&(tl.abs(bv)<1.e-6)
    else:
        beta_mask=tt<length
    tl.atomic_add(db+(bos+tt)*HV+ih,d_beta,mask=beta_mask,sem='relaxed')
    dk_add=de_add*bv[:,None]+dy_add
    dg_add=qv*dq_add+(de_add*bv[:,None]-dy_add)*kv
    if VARLEN: hc=tl.load(CO+seq).to(tl.int64)+it
    else: hc=(ibh//HV).to(tl.int64)*tl.cdiv(T,BT)+it
    # All shapes use V64. FP32 local sums and original operand casts remain.
    assert not COMPENSATE and not ENDPOINT
    oq=tl.zeros((BT,BK),tl.float32)
    dkfuture=tl.zeros((BT,BK),tl.float32)
    direct=tl.zeros((BT,BK),tl.float32)
    endpoint=tl.zeros((BK,),tl.float32)
    for iv in range(tl.cdiv(V,64)):
        vv=iv*64+tl.arange(0,64)
        if VF: so=vv[:,None]*K+kk[None,:]
        else: so=kk[None,:]*V+vv[:,None]
        smask=(vv[:,None]<V)&(kk[None,:]<K)
        hs=tl.load(HSTATE+(hc*HV+ih)*K*V+so,smask,0.)
        dhs=tl.load(DHSTATE+(hc*HV+ih)*K*V+so,smask,0.)
        endpoint+=tl.sum(hs.to(tl.float32)*dhs.to(tl.float32),0)
        vo=((bos+tt[:,None])*HV+ih)*V+vv[None,:]
        vm=(tt[:,None]<length)&(vv[None,:]<V)
        dov=tl.load(DOUT+vo,vm,0.)
        oq+=tl.dot(dov,hs.to(dov.dtype),allow_tf32=False)
        rv=tl.load(R+vo,vm,0.)
        dkfuture+=tl.dot(rv,dhs.to(rv.dtype),allow_tf32=False)
        pv=tl.load(PP+vo,vm,0.).to(rv.dtype)
        direct-=tl.dot(pv,hs.to(pv.dtype),allow_tf32=False)
    last=min((it+1)*BT,length)-1
    gn=tl.load(g+((bos+last)*HV+ih)*K+kk,kk<K,0.).to(tl.float32)
    gate=exp2(gv)
    gatefuture=exp2(gn[None,:]-gv)
    oq*=gate*scale
    dkfuture*=gatefuture
    ok=dkfuture+direct*bv[:,None]*gate
    endpoint=endpoint*exp2(gn)+tl.sum(kv*dkfuture,0)
    og=qv*oq-kv*dkfuture+kv*gate*direct*bv[:,None]+tl.where(tt[:,None]==last,endpoint[None,:],0.)
    db_direct=tl.sum(direct*kv*gate,1)
    if LOGIT_SCALE>0: db_direct*=bv*(1.-bv/LOGIT_SCALE)
    tl.atomic_add(db+(bos+tt)*HV+ih,db_direct,
                  mask=(tt<length)&(tl.abs(bv)<1.e-6),sem='relaxed')
    outg=og+dg_add
    if REVERSE_CUMSUM:
        outg=tl.cumsum(tl.where(valid,outg,0.),axis=0,reverse=True)
    tl.store(dq+gs,oq+dq_add,mask=valid)
    tl.store(dk+gs,ok+dk_add,mask=valid)
    tl.store(dg+gs,outg,mask=valid)
