"""Row-slab GDN consumer: bounded CxC gate/Gram live ranges on Ascend."""
from common import *
import triton.language as tl

@triton.jit
def consumer(K0,G,BETA,P,DM,H0,DK,DBP,DGP,CU,CI,
             T:tl.constexpr,B:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
             C:tl.constexpr,NK:tl.constexpr,NV:tl.constexpr,NR:tl.constexpr,VAR:tl.constexpr,
             GRAM:tl.constexpr,SVF:tl.constexpr,TASKS:tl.constexpr,STEP:tl.constexpr):
    for task in range(tl.program_id(0),TASKS,STEP):
        ik=task%tl.cdiv(K,NK);x=task//tl.cdiv(K,NK);ir=x%tl.cdiv(C,NR);x=x//tl.cdiv(C,NR)
        it=x//(B*HV);bh=x%(B*HV);ih=bh%HV
        if VAR:
            tg=it;seq=tl.load(CI+2*it);it=tl.load(CI+2*it+1);bos=tl.load(CU+seq);length=tl.load(CU+seq+1)-bos
        else:bos=(bh//HV)*T;length=T;tg=(bh//HV)*tl.cdiv(T,C)+it
        row=ir*NR+tl.arange(0,NR);col=tl.arange(0,C);tt=it*C+row;tc=it*C+col;kk=ik*NK+tl.arange(0,NK)
        off=(bos+tt)*HV+ih;alloff=(bos+tc)*HV+ih;ko=(bos+tt)*H+ih//(HV//H);kaoff=(bos+tc)*H+ih//(HV//H)
        pair=(tt[:,None]<length)&(tc[None,:]<length)
        gr=tl.load(G+off,tt<length,0).to(tl.float32);ga=tl.load(G+alloff,tc<length,0).to(tl.float32)
        br=tl.load(BETA+off,tt<length,0).to(tl.float32);ba=tl.load(BETA+alloff,tc<length,0).to(tl.float32)
        de=tl.zeros((NR,NK),tl.float32)
        for iv in range(tl.cdiv(V,NV)):
            vv=iv*NV+tl.arange(0,NV)
            p=tl.load(P+off[:,None]*V+vv[None,:],(tt[:,None]<length)&(vv[None,:]<V),0).to(K0.dtype.element_ty)
            if SVF:hp=(tg*HV+ih)*K*V+vv[:,None]*K+kk[None,:]
            else:hp=(tg*HV+ih)*K*V+kk[None,:]*V+vv[:,None]
            h=tl.load(H0+hp,(vv[:,None]<V)&(kk[None,:]<K),0).to(K0.dtype.element_ty)
            de-=tl.dot(p,h,allow_tf32=False)
        dd=tl.load(DM+off[:,None]*C+col[None,:],pair,0)
        dd=tl.where(row[:,None]>col[None,:],dd*tl.exp2(tl.where(row[:,None]>col[None,:],gr[:,None]-ga[None,:],0)),0).to(K0.dtype.element_ty)
        keys=tl.load(K0+kaoff[:,None]*K+kk[None,:],(tc[:,None]<length)&(kk[None,:]<K),0)
        left=tl.dot(dd,keys,allow_tf32=False)
        dt=tl.load(DM+alloff[None,:]*C+row[:,None],pair,0)
        dt=tl.where(col[None,:]>row[:,None],dt*tl.exp2(tl.where(col[None,:]>row[:,None],ga[None,:]-gr[:,None],0)),0).to(K0.dtype.element_ty)
        keys2=tl.load(K0+kaoff[:,None]*K+kk[None,:],(tc[:,None]<length)&(kk[None,:]<K),0)
        right=tl.dot(dt,(keys2*ba[:,None]).to(keys2.dtype),allow_tf32=False)
        kr=tl.load(K0+ko[:,None]*K+kk[None,:],(tt[:,None]<length)&(kk[None,:]<K),0).to(tl.float32)
        eg=tl.exp2(gr)
        dk=de*(br*eg)[:,None]+br[:,None]*left+right
        db=tl.sum(de*kr*eg[:,None]+left*kr,1)
        dg=tl.sum(de*kr*(br*eg)[:,None],1)
        if GRAM:
            kr2=tl.load(K0+ko[:,None]*K+kk[None,:],(tt[:,None]<length)&(kk[None,:]<K),0)
            kt=tl.load(K0+kaoff[None,:]*K+kk[:,None],(tc[None,:]<length)&(kk[:,None]<K),0)
            gram=tl.dot(kr2,kt,allow_tf32=False)
            dr=tl.load(DM+off[:,None]*C+col[None,:],pair,0)
            dr=tl.where(row[:,None]>col[None,:],dr*tl.exp2(tl.where(row[:,None]>col[None,:],gr[:,None]-ga[None,:],0)),0).to(K0.dtype.element_ty).to(tl.float32)
            dc=tl.load(DM+alloff[None,:]*C+row[:,None],pair,0)
            dc=tl.where(col[None,:]>row[:,None],dc*tl.exp2(tl.where(col[None,:]>row[:,None],ga[None,:]-gr[:,None],0)),0).to(K0.dtype.element_ty).to(tl.float32)
            dg+=tl.sum(gram*(dr*br[:,None]-dc*ba[None,:]),1)
        else:dg+=tl.sum(kr*(br[:,None]*left-right),1)
        tl.store(DK+off[:,None]*K+kk[None,:],dk,(tt[:,None]<length)&(kk[None,:]<K))
        tl.store(DBP+off*tl.cdiv(K,NK)+ik,db,tt<length);tl.store(DGP+off*tl.cdiv(K,NK)+ik,dg,tt<length)
