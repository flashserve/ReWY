"""Separate optional-interface acceptance, without performance claims."""
import argparse,gc
from common import *
from scalar_ops import ScalarOp
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--op',required=True);ap.add_argument('--output',required=True);ap.add_argument('--variants');a=ap.parse_args()
    if a.op=='gdn2':
        raise ValueError('Use the vector paper entry for GDN2')
        op=GDN2()
    elif a.op=='kda':
        raise ValueError('Use the vector paper entry for KDA')
        op=KDA()
    else:op=ScalarOp(a.op)
    variants=a.variants.split(',') if a.variants else op.variants
    path=HERE/'results'/a.output
    if path.exists():raise FileExistsError(path)
    emit(dict(type='environment',device=DEV,args=vars(a),hashes=hashes()),path)
    failed=[]
    for case_name in ('full_native','one_token','packed','absent_initial_final'):
        packed=case_name=='packed';absent=case_name=='absent_initial_final';lengths=[1,63,65,127] if packed else [256 if case_name=='full_native' else 1 if case_name=='one_token' else 65]
        ends=[0]
        for n in lengths:ends.append(ends[-1]+n)
        xs,dys=op.inputs(dict(B=1,T=ends[-1],H=2,D=128 if case_name=='full_native' else 64,V=128 if case_name=='full_native' else 96))
        if packed:
            h=torch.randn((len(lengths),*xs[-1].shape[1:]),device=DEV,dtype=torch.float32)*.1
            xs=(*xs[:-1],h);dys=(dys[0],torch.randn_like(h)*.1)
            from fla.ops.utils import prepare_chunk_indices
            cu=torch.tensor(ends,device=DEV,dtype=torch.int64)
            op.sequence=dict(cu_seqlens=cu,chunk_indices=prepare_chunk_indices(cu,64))
        else:op.sequence={}
        cpu=tuple(x.detach().cpu().double().requires_grad_() for x in xs)
        outs=[];states=[]
        for j,(start,end) in enumerate(zip(ends[:-1],ends[1:])):
            args=tuple(x[:,start:end] for x in cpu[:-1])+(torch.zeros_like(cpu[-1][j:j+1]) if absent else cpu[-1][j:j+1],)
            o,h=op.oracle(args);outs.append(o);states.append(h)
        out=torch.cat(outs,1),torch.cat(states,0)
        grad_dys=tuple(x.cpu().double() for x in dys)
        if absent:
            xs=(*xs[:-1],None);dys=(dys[0],None)
            ref=torch.autograd.grad(out[0],cpu[:-1],grad_dys[0]);names=op.names[:-1]
        else:ref=torch.autograd.grad(out,cpu,grad_dys);names=op.names
        for policy in ('recompute','all'):
            fw=op.forward(xs,policy)
            for v in variants:
                if hasattr(op,'reset_dispatch'):op.reset_dispatch()
                grads=op.backward(xs,dys,fw,v)
                if absent:
                    if grads[-1] is not None:raise RuntimeError('dh0 must be absent')
                    grads=grads[:-1]
                m=metrics(ref,grads,names);m.update(metrics(out,fw[2],['output','final_state']))
                emit(dict(type='correctness',case_name=case_name,policy=policy,variant=v,metrics=m,passed=check(m),dispatch=op.dispatch() if hasattr(op,'dispatch') else {}),path)
                if not check(m):failed.append((case_name,policy,v))
        del xs,dys,cpu,out,ref,fw;gc.collect();ACC.empty_cache()
    emit(dict(type='dependencies',**dependencies()),path)
    emit(dict(type='complete',passed=not failed,failures=failed),path)
    if failed:raise RuntimeError(str(failed))
if __name__=='__main__':
    with ACC.stream(ACC.Stream()):main()
