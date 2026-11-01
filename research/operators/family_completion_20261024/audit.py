"""Memory, stage attribution, and finite update consistency; not model convergence."""
import argparse,gc
from common import *
from scalar_ops import ScalarOp

def tensors(x):
    if isinstance(x,torch.Tensor):yield x
    elif isinstance(x,dict):
        for y in x.values():yield from tensors(y)
    elif isinstance(x,(tuple,list)):
        for y in x:yield from tensors(y)
def saved_bytes(fw,xs):
    ignored={x.untyped_storage().data_ptr() for x in tensors((xs,fw[2]))}
    stores={x.untyped_storage().data_ptr():x.untyped_storage().nbytes() for x in tensors(fw[:2])}
    return sum(v for k,v in stores.items() if k not in ignored)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--op',required=True);ap.add_argument('--output',required=True);ap.add_argument('--variants');ap.add_argument('--skip-stage',action='store_true');a=ap.parse_args()
    if a.op=='gdn2':
        from gdn2_op import GDN2
        op=GDN2()
    elif a.op=='kda':
        from kda_op import KDA
        op=KDA()
    else:op=ScalarOp(a.op)
    variants=a.variants.split(',') if a.variants else op.variants
    path=HERE/'results'/a.output
    if path.exists():raise FileExistsError(path)
    emit(dict(type='environment',device=DEV,args=vars(a),hashes=hashes()),path)
    for T in (2048,8192):
        case=dict(B=1,T=T,H=4 if T==2048 else 16,D=128);xs,dys=op.inputs(case)
        for policy in ('recompute','all'):
            fw=op.forward(xs,policy)
            for variant in variants:
                tmp=op.backward(xs,dys,fw,variant);ACC.synchronize();del tmp;gc.collect()
                before=ACC.memory_allocated();ACC.reset_peak_memory_stats()
                out=op.backward(xs,dys,fw,variant);ACC.synchronize()
                emit(dict(type='memory',case=case,policy=policy,variant=variant,saved_auxiliary_bytes=saved_bytes(fw,xs),backward_peak_extra_bytes=ACC.max_memory_allocated()-before,gradient_bytes=sum(x.nbytes for x in out)),path)
                del out
            if T==2048 and a.op!='kda' and not a.skip_stage:
                graphs={};held={};stage_functions={};op.record_stage=True
                for v in variants:
                    op.backward(xs,dys,fw,v)
                    # The closure owns intermediates allocated outside capture.
                    # Keep every variant's inputs alive for all graph replays.
                    stage_functions[v]=op.last_stage
                    graphs[v],held[v]=capture(stage_functions[v])
                emit(dict(type='stage_benchmark',boundary='WY' if a.op in ('delta','gdn') else 'WY_and_intra',policy=policy,case=case,**timing(graphs,10)),path)
                op.record_stage=False;op.last_stage=None;del graphs,held,stage_functions
            if T==2048:
                graphs={};held={}
                def public_backward(v):
                    return tuple(d.to(x.dtype) for d,x in zip(op.backward(xs,dys,fw,v),xs))
                reference=public_backward('original')
                for v in variants:
                    actual=public_backward(v);m=metrics(reference,actual,op.names)
                    if not check(m):raise RuntimeError('output-cast correctness failed '+v)
                    graphs[v],held[v]=capture(lambda v=v:public_backward(v))
                times=timing(graphs,10)
                for v,graph in graphs.items():
                    graph.replay();ACC.synchronize()
                    if not all(check(metrics(reference,z,op.names)) for z in held[v]):raise RuntimeError('output-cast replay failed '+v)
                emit(dict(type='public_dtype_benchmark',boundary='complete_backward_with_output_casts',case=case,policy=policy,passed=True,**times),path)
                del graphs,held,graph,reference,actual;gc.collect();ACC.empty_cache()
            del fw
        del xs,dys;gc.collect();ACC.empty_cache()
    case=dict(B=1,T=256,H=2,D=128);xs,dys=op.inputs(case)
    target=op.oracle(tuple(x.detach().cpu().double() for x in xs))
    # A fixed target and FP32 master inputs give a finite operator-level update test.
    target=tuple((x*.5).float().to(DEV) for x in target);dtypes=[x.dtype for x in xs]
    initial=[x.detach().float().clone() for x in xs]
    baseline=None
    for variant in variants:
        master=[x.clone() for x in initial];history=[]
        for step in range(5):
            cur=tuple(x.to(dt) for x,dt in zip(master,dtypes));fw=op.forward(cur,'all')
            out=fw[2];gradout=tuple((2*(o-t)/o.numel()*256*2).to(o.dtype) for o,t in zip(out,target))
            grad=op.backward(cur,gradout,fw,variant)
            loss=sum(((o.float()-t.float())**2).mean().item() for o,t in zip(out,target))
            master=[x-.01*d.float() for x,d in zip(master,grad)]
            history.append(loss)
        if baseline is None:baseline=tuple(x.clone() for x in master)
        m=metrics(baseline,master,op.names)
        emit(dict(type='finite_updates',variant=variant,steps=5,learning_rate=.01,loss=history,metrics=m,passed=check(m)),path)
        if not check(m):raise RuntimeError('finite update divergence')
    emit(dict(type='dependencies',**dependencies()),path)
    emit(dict(type='complete',passed=True),path)
if __name__=='__main__':
    with ACC.stream(ACC.Stream()):main()
