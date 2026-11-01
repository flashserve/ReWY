"""GDN residual with full-chunk ownership and bounded phase lifetimes.

P/dM cross the producer boundary. Direct key gradients are completed before
the established full-chunk matrix consumer; no 16-row/key partial task fanout.
Each stream owns private scratch. Public layouts and forward remain unchanged.
"""
from common import torch,triton,launch
import triton.language as tl
from gdn_producer_layout_probe import producer_scalar_layout as producer
import os

@triton.jit
def direct(K0,G,BETA,P,H0,DBV,DK,DB,DG,CU,CI,
           T:tl.constexpr,B:tl.constexpr,H:tl.constexpr,HV:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
           C:tl.constexpr,NK:tl.constexpr,NV:tl.constexpr,VAR:tl.constexpr,SVF:tl.constexpr,
           TASKS:tl.constexpr,STEP:tl.constexpr):
    for task in range(tl.program_id(0),TASKS,STEP):
        it=task//(B*HV);bh=task%(B*HV);ih=bh%HV
        if VAR:
            tg=it;seq=tl.load(CI+2*it).to(tl.int32);it=tl.load(CI+2*it+1).to(tl.int32)
            bos=tl.load(CU+seq);length=(tl.load(CU+seq+1)-bos).to(tl.int32)
        else:
            bos=(bh//HV)*T;length=T;tg=(bh//HV)*tl.cdiv(T,C)+it
        rr=tl.arange(0,C);tt=it*C+rr;off=(bos+tt)*HV+ih
        soff=(bh//HV)*HV*T+ih*T+tt if not VAR else ih*T+bos+tt
        beta=tl.load(BETA+soff,tt<length,0).to(tl.float32)
        eg=tl.exp2(tl.load(G+soff,tt<length,0).to(tl.float32))
        db=tl.load(DBV+soff,tt<length,0).to(tl.float32)
        dg=tl.zeros((C,),tl.float32)
        for ik in range(tl.cdiv(K,NK)):
            de=tl.zeros((C,NK),tl.float32)
            for iv in range(tl.cdiv(V,NV)):
                pp=tl.load(tl.make_block_ptr(P+(bos*HV+ih)*V,(length,V),(HV*V,1),(it*C,iv*NV),(C,NV),(1,0)),boundary_check=(0,1),padding_option='zero').to(K0.dtype.element_ty)
                if SVF:
                    hp=tl.make_block_ptr(H0+(tg*HV+ih)*K*V,(V,K),(K,1),(iv*NV,ik*NK),(NV,NK),(1,0))
                else:
                    hp=tl.make_block_ptr(H0+(tg*HV+ih)*K*V,(V,K),(1,V),(iv*NV,ik*NK),(NV,NK),(0,1))
                hh=tl.load(hp,boundary_check=(0,1),padding_option='zero').to(K0.dtype.element_ty)
                de-=tl.dot(pp,hh,allow_tf32=False)
            kp=tl.make_block_ptr(K0+(bos*H+ih//(HV//H))*K,(length,K),(H*K,1),(it*C,ik*NK),(C,NK),(1,0))
            key=tl.load(kp,boundary_check=(0,1),padding_option='zero').to(tl.float32)
            part=tl.sum(de*key,1)*eg
            db+=part;dg+=part*beta
            out=de*(beta*eg)[:,None]
            op=tl.make_block_ptr(DK+(bos*HV+ih)*K,(length,K),(HV*K,1),(it*C,ik*NK),(C,NK),(1,0))
            tl.store(op,out.to(DK.dtype.element_ty),boundary_check=(0,1))
        tl.store(DB+soff,db,tt<length);tl.store(DG+soff,dg,tt<length)

@triton.jit
def gate_transpose(DM,G,OUT,CU,CI,T:tl.constexpr,B:tl.constexpr,HV:tl.constexpr,C:tl.constexpr,
                   VAR:tl.constexpr,TASKS:tl.constexpr,STEP:tl.constexpr):
    for task in range(tl.program_id(0),TASKS,STEP):
        it=task//(B*HV);bh=task%(B*HV);ih=bh%HV
        if VAR:
            seq=tl.load(CI+2*it);it=tl.load(CI+2*it+1);bos=tl.load(CU+seq);length=tl.load(CU+seq+1)-bos
        else:bos=(bh//HV)*T;length=T
        rr=tl.arange(0,C);tt=it*C+rr;off=(bos+tt)*HV+ih
        soff=(bh//HV)*HV*T+ih*T+tt if not VAR else ih*T+bos+tt
        gg=tl.load(G+soff,tt<length,0).to(tl.float32)
        dm=tl.load(DM+off[None,:]*C+rr[:,None],(tt[:,None]<length)&(tt[None,:]<length),0)
        mask=(rr[None,:]>rr[:,None])&(tt[:,None]<length)&(tt[None,:]<length)
        out=tl.where(mask,dm*tl.exp2(tl.where(mask,gg[None,:]-gg[:,None],0)),0)
        tl.store(OUT+off[:,None]*C+rr[None,:],out,tt[:,None]<length)

def wy(k,v,beta,A,du,h,r,g=None,gram=False,state_v_first=False,cu_seqlens=None,chunk_indices=None,**unused):
    from fla.ops.utils import prepare_chunk_indices
    import gdn_gram_npu as no_gram
    import gdn_original_separate_npu as old_gram
    B,T,H,K=k.shape;HV=v.shape[2];V=v.shape[-1];C=A.shape[-1]
    if cu_seqlens is not None and chunk_indices is None:chunk_indices=prepare_chunk_indices(cu_seqlens,C)
    nt=len(chunk_indices) if cu_seqlens is not None else triton.cdiv(T,C)
    nk=int(os.environ.get('GDN_NK','128'));nv=int(os.environ.get('GDN_NV','128'))
    p=torch.empty_like(v,dtype=torch.float32);dm=torch.empty_like(A,dtype=torch.float32);dt=torch.empty_like(dm)
    dv=torch.empty_like(v);dbv=torch.empty((B,HV,T),device=k.device,dtype=torch.float32)
    dk=torch.empty((B,T,HV,K),device=k.device,dtype=k.dtype)
    db=torch.empty((B,HV,T),device=k.device,dtype=beta.dtype);dg=torch.empty((B,HV,T),device=k.device,dtype=g.dtype)
    gc=g.transpose(1,2).contiguous();bc=beta.transpose(1,2).contiguous()
    shared=dict(T=T,B=B,HV=HV,C=C,VAR=cu_seqlens is not None,CU=cu_seqlens,CI=chunk_indices)
    launch(producer,nt*B*HV,A=A,DR=du,R=r,V0=v,BETA=bc,P=p,DV=dv,DBV=dbv,DM=dm,NV=nv,V=V,LOAD_CONTIG=True,STORE_CONTIG=True,**shared)
    launch(direct,nt*B*HV,K0=k,G=gc,BETA=bc,P=p,H0=h,DBV=dbv,DK=dk,DB=db,DG=dg,H=H,K=K,V=V,NK=nk,NV=nv,SVF=state_v_first,**shared)
    launch(gate_transpose,nt*B*HV,DM=dm,G=gc,OUT=dt,**shared)
    module=old_gram if gram else no_gram
    base=dict(cu_seqlens=cu_seqlens,chunk_indices=chunk_indices,T=T,BT=C,IS_VARLEN=cu_seqlens is not None)
    fin=dict(k=k,beta=bc,dA_out=dt,dk=dk,db=db,H=H,HV=HV,K=K,BK=min(128,triton.next_power_of_2(K)),
             BETA_T_CONTIG=True,DB_T_CONTIG=True,B=B,**base)
    if not gram:fin.update(dg=dg,USE_G=True,DG_T_CONTIG=True)
    module._launch_wy_core_grid(module.prepare_wy_repr_bwd_finalize_k_npu,task_num=nt*B*HV,kernel_kwargs=fin)
    if gram:
        module._launch_wy_kernel(module.prepare_wy_repr_bwd_finalize_a2_dg_npu,NT=nt,bh_total=B*HV,
            kernel_kwargs=dict(k=k,beta=bc,dA_out=dt,dg=dg,H=H,HV=HV,K=K,BK=min(128,triton.next_power_of_2(K)),
                BETA_T_CONTIG=True,DG_T_CONTIG=True,**base))
    if H!=HV:dk=dk.reshape(B,T,H,HV//H,K).sum(3)
    return dk,dv,db.transpose(1,2).contiguous(),dg.transpose(1,2).contiguous()
