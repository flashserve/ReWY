"""Bounded-UB whole-chunk producer, optional compensated P and beta-logit VJP."""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['T'])
def full_producer(v,r,A,dr,beta,P,dv,db,dt,cu,ci,T,HV:tl.constexpr,V:tl.constexpr,
                  VP:tl.constexpr,BT:tl.constexpr,VARLEN:tl.constexpr,
                  MATERIALIZED:tl.constexpr,COMPENSATE:tl.constexpr,LOGIT_SCALE:tl.constexpr):
    it,ibh=tl.program_id(0),tl.program_id(1)
    ih=ibh%HV
    if VARLEN:
        seq=tl.load(ci+2*it).to(tl.int32)
        it=tl.load(ci+2*it+1).to(tl.int32)
        bos=tl.load(cu+seq).to(tl.int64)
        length=tl.load(cu+seq+1).to(tl.int64)-bos
    else:
        bos=(ibh//HV).to(tl.int64)*T
        length=T
    rr=tl.arange(0,BT)
    vv=tl.arange(0,VP)
    tt=it*BT+rr
    off=(bos+tt)*HV+ih
    aa=tl.load(A+off[None,:]*BT+rr[:,None],mask=(tt[None,:]<length)&(tt[:,None]<length),other=0.)
    value=tl.load(dr+off[:,None]*V+vv[None,:],mask=(tt[:,None]<length)&(vv[None,:]<V),other=0.)
    pp=tl.dot(aa,value,allow_tf32=False)
    rv=tl.load(r+off[:,None]*V+vv[None,:],mask=(tt[:,None]<length)&(vv[None,:]<V),other=0.)
    low=pp.to(rv.dtype)
    bb=tl.load(beta+off,mask=tt<length,other=0.).to(tl.float32)
    fallback=(tl.abs(bb)<1.e-6)&(tt<length)
    dotpr=tl.sum(pp*rv,1)
    if LOGIT_SCALE>0:
        dbeta=dotpr*(1.-bb/LOGIT_SCALE)
    else:
        dbeta=dotpr/tl.where(fallback | (tt>=length),1.,bb)
    if tl.sum(fallback.to(tl.int32),0)>0:
        orig=tl.load(v+off[:,None]*V+vv[None,:],mask=(tt[:,None]<length)&(vv[None,:]<V),other=0.)
        direct=tl.sum(pp*orig,1)
        if LOGIT_SCALE>0: direct*=bb*(1.-bb/LOGIT_SCALE)
        dbeta=tl.where(fallback,direct,dbeta)
    vm=(tt[:,None]<length)&(vv[None,:]<V)
    tl.store(P+off[:,None]*V+vv[None,:],pp if COMPENSATE else low,mask=vm)
    tl.store(dv+off[:,None]*V+vv[None,:],pp*bb[:,None],mask=vm)
    tl.store(db+off,dbeta,mask=tt<length)
    if MATERIALIZED:
        d=-tl.dot(low,tl.trans(rv),allow_tf32=False)
        if COMPENSATE:
            rem=(pp-low.to(tl.float32)).to(low.dtype)
            d-=tl.dot(rem,tl.trans(rv),allow_tf32=False)
        d=tl.where(rr[:,None]>rr[None,:],d,0.)
        tl.store(dt+off[:,None]*BT+rr[None,:],d,mask=tt[:,None]<length)
