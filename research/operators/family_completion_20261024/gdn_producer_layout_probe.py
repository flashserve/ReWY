"""Isolate scalar tensor layout; all matrix equations/loads and tile sizes unchanged."""
from common import triton
import triton.language as tl
@triton.jit
def producer_scalar_layout(A, DR, R, V0, BETA, P, DV, DBV, DM, CU, CI,
             T:tl.constexpr,B:tl.constexpr,HV:tl.constexpr,V:tl.constexpr,
             C:tl.constexpr,NV:tl.constexpr,VAR:tl.constexpr,TASKS:tl.constexpr,STEP:tl.constexpr,VECTOR:tl.constexpr=False,LOAD_CONTIG:tl.constexpr=False,STORE_CONTIG:tl.constexpr=False):
    for task in range(tl.program_id(0),TASKS,STEP):
        it=task//(B*HV); bh=task%(B*HV); ih=bh%HV
        if VAR:
            seq=tl.load(CI+2*it); it=tl.load(CI+2*it+1)
            bos=tl.load(CU+seq); length=tl.load(CU+seq+1)-bos
        else:
            bos=(bh//HV)*T; length=T
        rr=tl.arange(0,C); tt=it*C+rr; off=(bos+tt)*HV+ih
        dd=tl.zeros((C,C),tl.float32); db=tl.zeros((C,),tl.float32)
        soff=(bh//HV)*HV*T+ih*T+tt if not VAR else ih*T+bos+tt
        load_off=soff if LOAD_CONTIG else off
        store_off=soff if STORE_CONTIG else off
        if not VECTOR:beta=tl.load(BETA+load_off,tt<length,other=0).to(tl.float32)
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
        if not VECTOR:tl.store(DBV+store_off,db,tt<length)
