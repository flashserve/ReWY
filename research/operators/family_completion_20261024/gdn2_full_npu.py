"""Transfer residual-WY on-chip P reuse to independent erase/write gates.

No P is quantized for storage: P has no global storage at all. Its FP32
accumulator produces dv/dw, and dot operands use the same BF16 conversion
as the accepted GDN2 consumer. dM and all partial gradients remain FP32.
"""
from common import *
import triton.language as tl

@triton.jit
def core(Q,K0,G,B0,V0,WG,A,DR,R,H0,DH,DO,DQ,DK,DB,DG,DV,DW,DM,CU,CI,
         T:tl.constexpr,B:tl.constexpr,H:tl.constexpr,K:tl.constexpr,V:tl.constexpr,C:tl.constexpr,
         NK:tl.constexpr,VP:tl.constexpr,SCALE:tl.constexpr,VAR:tl.constexpr,TASKS:tl.constexpr,STEP:tl.constexpr):
    for task in range(tl.program_id(0),TASKS,STEP):
        it=task//(B*H);bh=task%(B*H);ih=bh%H
        if VAR:
            tg=it;seq=tl.load(CI+2*it).to(tl.int32);it=tl.load(CI+2*it+1).to(tl.int32);bos=tl.load(CU+seq);length=(tl.load(CU+seq+1)-bos).to(tl.int32)
        else:bos=(bh//H)*T;length=T;tg=(bh//H)*tl.cdiv(T,C)+it
        rr=tl.arange(0,C);tt=it*C+rr;off=(bos+tt)*H+ih
        at=tl.load(A+off[None,:]*C+rr[:,None],(tt[:,None]<length)&(tt[None,:]<length),0)
        dr=tl.load(tl.make_block_ptr(DR+(bos*H+ih)*V,(length,V),(H*V,1),(it*C,0),(C,VP),(1,0)),boundary_check=(0,1),padding_option='zero')
        pp=tl.dot(at,dr,allow_tf32=False)
        value=tl.load(tl.make_block_ptr(V0+(bos*H+ih)*V,(length,V),(H*V,1),(it*C,0),(C,VP),(1,0)),boundary_check=(0,1),padding_option='zero')
        wg=tl.load(tl.make_block_ptr(WG+(bos*H+ih)*V,(length,V),(H*V,1),(it*C,0),(C,VP),(1,0)),boundary_check=(0,1),padding_option='zero')
        tl.store(tl.make_block_ptr(DV+(bos*H+ih)*V,(length,V),(H*V,1),(it*C,0),(C,VP),(1,0)),(pp*wg).to(DV.dtype.element_ty),boundary_check=(0,1))
        tl.store(tl.make_block_ptr(DW+(bos*H+ih)*V,(length,V),(H*V,1),(it*C,0),(C,VP),(1,0)),pp*value,boundary_check=(0,1))
        r=tl.load(tl.make_block_ptr(R+(bos*H+ih)*V,(length,V),(H*V,1),(it*C,0),(C,VP),(1,0)),boundary_check=(0,1),padding_option='zero')
        pdot=pp.to(r.dtype)
        dm=-tl.dot(pdot+0.,tl.trans(r),allow_tf32=False)
        dm=tl.where((tt[:,None]<length)&(tt[None,:]<length)&(rr[:,None]>rr[None,:]),dm,0)
        tl.store(DM+off[:,None]*C+rr[None,:],dm,tt[:,None]<length)
        for ik in range(tl.cdiv(K,NK)):
            hp=tl.make_block_ptr(H0+(tg*H+ih)*K*V,(V,K),(1,V),(0,ik*NK),(VP,NK),(0,1))
            dhp=tl.make_block_ptr(DH+(tg*H+ih)*K*V,(V,K),(1,V),(0,ik*NK),(VP,NK),(0,1))
            hs=tl.load(hp,boundary_check=(0,1),padding_option='zero')
            dhs=tl.load(dhp,boundary_check=(0,1),padding_option='zero')
            endpoint=tl.sum(hs*dhs,0)
            dout=tl.load(tl.make_block_ptr(DO+(bos*H+ih)*V,(length,V),(H*V,1),(it*C,0),(C,VP),(1,0)),boundary_check=(0,1),padding_option='zero')
            dq=tl.dot(dout,hs.to(dout.dtype),allow_tf32=False)
            rv=tl.load(tl.make_block_ptr(R+(bos*H+ih)*V,(length,V),(H*V,1),(it*C,0),(C,VP),(1,0)),boundary_check=(0,1),padding_option='zero')
            dk=tl.dot(rv,dhs.to(rv.dtype),allow_tf32=False)
            hs2=tl.load(hp,boundary_check=(0,1),padding_option='zero').to(K0.dtype.element_ty)
            de=-tl.dot(pdot+0.,hs2,allow_tf32=False)
            kp=tl.make_block_ptr(K0+(bos*H+ih)*K,(length,K),(H*K,1),(it*C,ik*NK),(C,NK),(1,0))
            qp=tl.make_block_ptr(Q+(bos*H+ih)*K,(length,K),(H*K,1),(it*C,ik*NK),(C,NK),(1,0))
            gp=tl.make_block_ptr(G+(bos*H+ih)*K,(length,K),(H*K,1),(it*C,ik*NK),(C,NK),(1,0))
            bp=tl.make_block_ptr(B0+(bos*H+ih)*K,(length,K),(H*K,1),(it*C,ik*NK),(C,NK),(1,0))
            key=tl.load(kp,boundary_check=(0,1),padding_option='zero').to(tl.float32)
            query=tl.load(qp,boundary_check=(0,1),padding_option='zero').to(tl.float32)
            gg=tl.load(gp,boundary_check=(0,1),padding_option='zero').to(tl.float32)
            bb=tl.load(bp,boundary_check=(0,1),padding_option='zero').to(tl.float32)
            kk=ik*NK+tl.arange(0,NK)
            gn=tl.load(G+((bos+tl.minimum((it+1)*C,length)-1)*H+ih)*K+kk,kk<K,0).to(tl.float32)
            eg=tl.exp2(gg);dq*=eg*SCALE;dk*=tl.exp2(gn[None,:]-gg)
            endpoint=endpoint*tl.exp2(gn)+tl.sum(dk*key,0)
            db=de*key*eg
            dg=query*dq-key*dk+db*bb+(tt==tl.minimum((it+1)*C,length)-1)[:,None]*endpoint[None,:]
            dk+=de*eg*bb
            tl.store(tl.make_block_ptr(DQ+(bos*H+ih)*K,(length,K),(H*K,1),(it*C,ik*NK),(C,NK),(1,0)),dq,boundary_check=(0,1))
            tl.store(tl.make_block_ptr(DK+(bos*H+ih)*K,(length,K),(H*K,1),(it*C,ik*NK),(C,NK),(1,0)),dk,boundary_check=(0,1))
            tl.store(tl.make_block_ptr(DB+(bos*H+ih)*K,(length,K),(H*K,1),(it*C,ik*NK),(C,NK),(1,0)),db,boundary_check=(0,1))
            tl.store(tl.make_block_ptr(DG+(bos*H+ih)*K,(length,K),(H*K,1),(it*C,ik*NK),(C,NK),(1,0)),dg,boundary_check=(0,1))

def residual(q,k,v,v_new,g,b,w_gate,A,h,do,dh,dv,scale=None,cu_seqlens=None,chunk_size=64,chunk_indices=None,state_v_first=False):
    from fla.ops.utils import prepare_chunk_indices
    B,T,H,K=k.shape;V=v.shape[-1];C=chunk_size
    if cu_seqlens is not None and chunk_indices is None:chunk_indices=prepare_chunk_indices(cu_seqlens,C)
    nt=len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T,C)
    dq=torch.empty_like(q,dtype=torch.float32);dk=torch.empty_like(k,dtype=torch.float32)
    db=torch.empty_like(b,dtype=torch.float32);dg=torch.empty_like(g,dtype=torch.float32)
    dw=torch.empty_like(w_gate,dtype=torch.float32);dv2=torch.empty_like(v);dm=torch.empty_like(A,dtype=torch.float32)
    shared=dict(T=T,B=B,H=H,K=K,V=V,C=C,VAR=cu_seqlens is not None,CU=cu_seqlens,CI=chunk_indices)
    hkey=h.transpose(-1,-2).contiguous() if state_v_first else h
    dhkey=dh.transpose(-1,-2).contiguous() if state_v_first else dh
    launch(core,nt*B*H,Q=q,K0=k,G=g,B0=b,V0=v,WG=w_gate,A=A,DR=dv,R=v_new,H0=hkey,DH=dhkey,DO=do,DQ=dq,DK=dk,DB=db,DG=dg,DV=dv2,DW=dw,DM=dm,
        NK=64,VP=triton.next_power_of_2(V),SCALE=scale,**shared)
    return dq,dk,dv2,db,dw,dg,dm
