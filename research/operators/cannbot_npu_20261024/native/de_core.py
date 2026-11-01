"""Isolated residual DE boundary: one P, direct DE output, no dk/dG RMW."""
import triton
import triton.language as tl
from fla.ops.utils.op import exp2


@triton.jit(do_not_specialize=['T','task_num','num_core','BH'])
def residual_de_core(v, residual, beta, A, dv, dv2, dA_acc, db_acc,
                      h,k,g,dk,dg,de_out,cu_seqlens,chunk_indices,chunk_offsets,
                      T,BH,task_num,num_core,H:tl.constexpr,HV:tl.constexpr,
                      K:tl.constexpr,V:tl.constexpr,BT:tl.constexpr,BV:tl.constexpr,
                      VP:tl.constexpr,BK:tl.constexpr,IS_VARLEN:tl.constexpr,
                      G_T_CONTIG:tl.constexpr,USE_GRAPH:tl.constexpr=False):
    core=tl.program_id(0)
    total_t=T
    for task in tl.range(core,task_num,num_core):
        it=task//BH
        ibh=task%BH
        ib,ih=ibh//HV,ibh%HV
        ikh=ih//(HV//H)
        valid=True
        if IS_VARLEN:
            seq=tl.load(chunk_indices+2*it).to(tl.int32)
            it=tl.load(chunk_indices+2*it+1).to(tl.int32)
            if USE_GRAPH: valid=seq>=0
        if valid:
            if IS_VARLEN:
                bos=tl.load(cu_seqlens+seq).to(tl.int64)
                T=(tl.load(cu_seqlens+seq+1)-bos).to(tl.int32)
                itg=tl.load(chunk_offsets+seq).to(tl.int64)+it
            else:
                bos=ib.to(tl.int64)*T
                itg=ib.to(tl.int64)*tl.cdiv(T,BT)+it
            if G_T_CONTIG:
                if IS_VARLEN: off=ih.to(tl.int64)*total_t+bos
                else: off=ibh.to(tl.int64)*total_t
                vs=V
                ats=BT
                bs=1
            else:
                off=bos*HV+ih
                vs=HV*V
                ats=HV*BT
                bs=HV
            av=tl.make_block_ptr(A+off*BT,(BT,T),(1,ats),(0,it*BT),(BT,BT),(0,1))
            dvr=tl.make_block_ptr(dv+off*V,(T,V),(vs,1),(it*BT,0),(BT,VP),(1,0))
            rr=tl.make_block_ptr(residual+off*V,(T,V),(vs,1),(it*BT,0),(BT,VP),(1,0))
            vr=tl.make_block_ptr(v+off*V,(T,V),(vs,1),(it*BT,0),(BT,VP),(1,0))
            bp=tl.make_block_ptr(beta+off,(T,),(bs,),(it*BT,),(BT,),(0,))
            a=tl.load(av,boundary_check=(0,1),padding_option='zero')
            dr=tl.load(dvr,boundary_check=(0,1),padding_option='zero')
            r=tl.load(rr,boundary_check=(0,1),padding_option='zero')
            b=tl.load(bp,boundary_check=(0,),padding_option='zero')
            pp=tl.dot(a,dr,allow_tf32=False)
            p=pp.to(r.dtype)
            value=tl.load(vr,boundary_check=(0,1),padding_option='zero')
            db=tl.sum(pp*value,1)
            dvout=tl.make_block_ptr(dv2+(bos*HV+ih)*V,(T,V),(HV*V,1),(it*BT,0),(BT,VP),(1,0))
            tl.store(dvout,(pp*b[:,None]).to(dvout.dtype.element_ty),boundary_check=(0,1))
            p_copy=p+0.0
            dt=tl.dot(p_copy,tl.trans(r),allow_tf32=False)
            rows=it*BT+tl.arange(0,BT)
            dt=tl.where((rows[:,None]>rows[None,:])&(rows[:,None]<T)&(rows[None,:]<T),-dt,0.)
            dtout=tl.make_block_ptr(dA_acc+(bos*HV+ih)*BT,(T,BT),(HV*BT,1),(it*BT,0),(BT,BT),(1,0))
            tl.store(dtout,dt,boundary_check=(0,1))
            for ik in range(tl.cdiv(K,BK)):
                hp=tl.make_block_ptr(h+(itg*HV+ih)*K*V,(V,K),(1,V),(0,ik*BK),(VP,BK),(0,1))
                hs=tl.load(hp,boundary_check=(0,1),padding_option='zero')
                pc=p+0.0
                de=-tl.dot(pc,hs.to(p.dtype),allow_tf32=False)
                dep=tl.make_block_ptr(de_out+(bos*HV+ih)*K,(T,K),(HV*K,1),(it*BT,ik*BK),(BT,BK),(1,0))
                tl.store(dep,de,boundary_check=(0,1))
            dbout=tl.make_block_ptr(db_acc+bos*HV+ih,(T,),(HV,),(it*BT,),(BT,),(0,))
            tl.store(dbout,db,boundary_check=(0,))
