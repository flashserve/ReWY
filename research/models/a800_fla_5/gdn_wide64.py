"""Separate direct adjoint from residual WY to bound V-wide live ranges.
No DW is materialized: direct core runs USE_DW=False; residual uses P=A^T dR.
Original BF16 dK partial rounding is restored before their final sum.
Original FP32 state/gate/accumulators and P's FP32 dV/db uses are unchanged.
"""
from common import torch,triton
import triton.language as tl
from fla.ops.common.chunk_o import chunk_bwd_kernel_dqkwg
from triton.runtime.jit import JITFunction
DIRECT=chunk_bwd_kernel_dqkwg
while not isinstance(DIRECT,JITFunction):DIRECT=DIRECT.fn

@triton.jit
def residual(K0,V0,G,BETA,A,DR,R,H0,DK0,DG0,DK,DV,DB,DG,CU,CI,
             T:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
             C:tl.constexpr,NK:tl.constexpr,NV:tl.constexpr,VAR:tl.constexpr,SVF:tl.constexpr,NP:tl.constexpr,RP:tl.constexpr,ALL:tl.constexpr):
 it=tl.program_id(0);bh=tl.program_id(1);ih=bh%HV
 if VAR:
  tg=it;seq=tl.load(CI+2*it);it=tl.load(CI+2*it+1)
  bos=tl.load(CU+seq);length=tl.load(CU+seq+1)-bos
 else:bos=(bh//HV)*T;length=T;tg=(bh//HV)*tl.cdiv(T,C)+it
 rr=tl.arange(0,C);tt=it*C+rr;off=(bos+tt)*HV+ih
 ko=(bos+tt)*H+ih//(HV//H);vv=tl.arange(0,NV)
 valid=tt<length;pair=valid[:,None]&valid[None,:];vm=valid[:,None]&(vv[None,:]<V)
 at=tl.load(A+off[None,:]*C+rr[:,None],pair,0)
 beta=tl.load(BETA+off,valid,0).to(tl.float32)
 db=tl.zeros((C,),tl.float32)
 pp=tl.full((C,NV),0,K0.dtype.element_ty)
 for iv in range(tl.cdiv(V,64)):
  vi=iv*64+tl.arange(0,64);mi=valid[:,None]&(vi[None,:]<V)
  dri=tl.load(DR+off[:,None]*V+vi[None,:],mi,0)
  p=tl.dot(at,dri,allow_tf32=False)
  value=tl.load(V0+off[:,None]*V+vi[None,:],mi,0)
  db+=tl.sum(p*value,1)
  tl.store(DV+off[:,None]*V+vi[None,:],p*beta[:,None],mi)
  small=p.to(K0.dtype.element_ty)
  expanded=tl.broadcast_to(small[:,None,:],(C,NV//64,64)).reshape(C,NV)
  pp=tl.where(vv[None,:]//64==iv,expanded,pp)
 r=tl.load(R+off[:,None]*V+vv[None,:],vm,0)
 dm=-tl.dot(pp,tl.trans(r),allow_tf32=False)
 g=tl.load(G+off,valid,0).to(tl.float32);eg=tl.exp2(g)
 fac=tl.exp2(tl.where(pair&(rr[:,None]>rr[None,:]),g[:,None]-g[None,:],0))
 dm=tl.where(pair&(rr[:,None]>rr[None,:]),dm*fac,0).to(K0.dtype.element_ty)
 ii=tl.arange(0,RP)
 dg=tl.sum(tl.load(DG0+ii[:,None]*ALL+off[None,:],(ii[:,None]<NP)&valid[None,:],0),0)
 for ik in range(tl.cdiv(K,NK)):
  kk=ik*NK+tl.arange(0,NK);km=valid[:,None]&(kk[None,:]<K)
  key=tl.load(K0+ko[:,None]*K+kk[None,:],km,0)
  if SVF:hp=(tg*HV+ih)*K*V+vv[:,None]*K+kk[None,:]
  else:hp=(tg*HV+ih)*K*V+kk[None,:]*V+vv[:,None]
  hm=(vv[:,None]<V)&(kk[None,:]<K)
  hs=tl.load(H0+hp,hm,0).to(key.dtype)
  erase=-tl.dot(pp,hs,allow_tf32=False)*eg[:,None]+tl.dot(dm,key,allow_tf32=False)
  piece=tl.sum(erase*key,1);db+=piece;dg+=beta*piece
  right=tl.dot(tl.trans(dm),(key*beta[:,None]).to(key.dtype),allow_tf32=False)
  dg-=tl.sum(right*key,1)
  dk_res=(erase*beta[:,None]+right).to(DK.dtype.element_ty)
  direct=tl.load(DK0+off[:,None]*K+kk[None,:],km,0)
  tl.store(DK+off[:,None]*K+kk[None,:],direct.to(tl.float32)+dk_res.to(tl.float32),km)
 tl.store(DB+off,db,valid)
 dg=tl.cumsum(tl.where(valid,dg,0),0,reverse=True)
 tl.store(DG+off,dg,valid)

def adjoint(q,k,v,g,beta,A,dr,r,h,dh,do,scale,state_v_first=False,cu_seqlens=None,chunk_indices=None):
 from fla.ops.utils import prepare_chunk_indices
 assert g is not None
 B,T,H,K=k.shape;HV=v.shape[2];V=v.shape[-1];C=A.shape[-1]
 assert H==HV,'prototype requires equal query/value heads'
 if cu_seqlens is not None and chunk_indices is None:chunk_indices=prepare_chunk_indices(cu_seqlens,C)
 nt=len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T,C)
 np=triton.cdiv(K,128)
 dq=torch.empty_like(q);dk0=torch.empty_like(k);dg0=torch.empty((np,B,T,HV),device=g.device,dtype=torch.float32)
 DIRECT[(np*nt,B*HV)](q=q,k=k,v=r,g=g,g_gamma=None,h=h,do=do,dh=dh,dq=dq,dk=dk0,dw=None,dv=None,dg=dg0,
    cu_seqlens=cu_seqlens,chunk_indices=chunk_indices,scale=scale,B=B,T=T,H=H,HV=HV,K=K,V=V,BT=C,BK=128,BV=64,
    USE_G=True,USE_G_GAMMA=False,USE_DW=False,STATE_V_FIRST=state_v_first,IS_VARLEN=cu_seqlens is not None,num_warps=8,num_stages=1)
 dk=torch.empty_like(dk0);dv=torch.empty_like(v);db=torch.empty_like(beta);dg=torch.empty_like(g)
 residual[(nt,B*HV)](k,v,g,beta,A,dr,r,h,dk0,dg0,dk,dv,db,dg,cu_seqlens,chunk_indices,
       NP=np,RP=triton.next_power_of_2(np),ALL=B*T*HV,T=T,H=H,HV=HV,K=K,V=V,C=C,NK=32,NV=max(64,triton.next_power_of_2(V)),VAR=cu_seqlens is not None,SVF=state_v_first,num_warps=4,num_stages=1)
 return dq,dk,dv,dg,db
