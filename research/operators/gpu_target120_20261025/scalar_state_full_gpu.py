"""Same serial state adjoint with a single padded K contraction.

K padding is purely dimensional. BV32/4warps/1stage applies to every shape.
State storage and accumulation and the gated FP32 Q operand are unchanged.
"""
from common import torch, triton
import triton.language as tl
from fla.ops.utils.op import exp2


@triton.jit
def state_kernel(Q,K0,W,G,DHT,DH0,DO,DH,DV,DR,CU,CO,
                 T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
                 C:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr,SCALE:tl.constexpr,
                 GATED:tl.constexpr,VAR:tl.constexpr,SVF:tl.constexpr,
                 INITIAL:tl.constexpr,FINAL:tl.constexpr):
    iv=tl.program_id(0);nh=tl.program_id(1);seq=nh//HV;ih=nh%HV
    if VAR:
        bos=tl.load(CU+seq);length=tl.load(CU+seq+1)-bos;boh=tl.load(CO+seq)
    else:
        bos=seq*T;length=T;boh=seq*tl.cdiv(T,C)
    kk=tl.arange(0,BK);vv=iv*BV+tl.arange(0,BV);rr=tl.arange(0,C)
    hm=(kk[:,None]<K)&(vv[None,:]<V)
    if SVF:so=vv[None,:]*K+kk[:,None]
    else:so=kk[:,None]*V+vv[None,:]
    state=tl.zeros((BK,BV),tl.float32)
    if FINAL:state+=tl.load(DHT+nh*K*V+so,hm,0)
    for it in range(tl.cdiv(length,C)-1,-1,-1):
        tt=it*C+rr;valid=tt<length
        off=(bos+tt)*HV+ih;ko=(bos+tt)*H+ih//(HV//H)
        tl.store(DH+((boh+it)*HV+ih)*K*V+so,state,hm)
        key=tl.load(K0+ko[:,None]*K+kk[None,:],valid[:,None]&(kk[None,:]<K),0)
        dr=tl.dot(key,state.to(key.dtype))
        if GATED:
            g=tl.load(G+off,valid,0).to(tl.float32)
            gn=tl.load(G+(bos+tl.minimum((it+1)*C,length)-1)*HV+ih).to(tl.float32)
            dr*=tl.where(valid,exp2(gn-g),0)[:,None]
        vm=valid[:,None]&(vv[None,:]<V)
        dr+=tl.load(DV+off[:,None]*V+vv[None,:],vm,0)
        tl.store(DR+off[:,None]*V+vv[None,:],dr,vm)
        wt=tl.load(W+off[None,:]*K+kk[:,None],(kk[:,None]<K)&valid[None,:],0)
        qt=tl.load(Q+ko[None,:]*K+kk[:,None],(kk[:,None]<K)&valid[None,:],0)
        dout=tl.load(DO+off[:,None]*V+vv[None,:],vm,0)
        if GATED:
            state*=exp2(gn)
            qt=qt*exp2(g)[None,:]
        # Match frozen reference: gated qt/dout dot has FP32 operands;
        # ungated dot and W/DR dot have the original input operand dtype.
        state+=tl.dot(qt,dout.to(qt.dtype))*SCALE-tl.dot(wt,dr.to(wt.dtype))
    if INITIAL:tl.store(DH0+nh*K*V+so,state,hm)


def state_bwd(q,k,w,do,dv,g=None,gk=None,h0=None,dht=None,scale=None,
              state_v_first=False,cu_seqlens=None,chunk_size=64,
              chunk_indices=None,chunk_offsets=None,value_block=32,**kw):
    from fla.ops.utils import prepare_chunk_indices,prepare_chunk_offsets
    assert gk is None
    B,T,H,K=q.shape;V=do.shape[-1];HV=do.shape[2];C=chunk_size
    if cu_seqlens is None:N=B;nt=triton.cdiv(T,C)
    else:
        N=len(cu_seqlens)-1
        if chunk_indices is None:chunk_indices=prepare_chunk_indices(cu_seqlens,C)
        nt=len(chunk_indices)
        if chunk_offsets is None:chunk_offsets=prepare_chunk_offsets(cu_seqlens,C)
    dims=(V,K) if state_v_first else (K,V)
    dh=torch.empty((B,nt,HV,*dims),device=q.device,dtype=q.dtype)
    dh0=torch.empty_like(h0,dtype=torch.float32) if h0 is not None else None
    dr=torch.empty_like(dv)
    state_kernel[(triton.cdiv(V,value_block),N*HV)](q,k,w,g,dht,dh0,do,dh,dv,dr,cu_seqlens,chunk_offsets,
        T=T,H=H,HV=HV,K=K,V=V,C=C,BK=triton.next_power_of_2(K),BV=value_block,SCALE=scale,
        GATED=g is not None,VAR=cu_seqlens is not None,SVF=state_v_first,
        INITIAL=h0 is not None,FINAL=dht is not None,num_warps=4,num_stages=1)
    return dh,dh0,dr
