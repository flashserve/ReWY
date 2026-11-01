"""Complete-B memory, public dtype, replay and finite-update acceptance.

No forward, forward+backward, or public-wrapper performance timing.
"""
import argparse
import gc
from common import *
from scalar_ops import ScalarOp
from audit import saved_bytes


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--op',required=True)
    ap.add_argument('--variants',required=True);ap.add_argument('--output',required=True)
    args=ap.parse_args();variants=args.variants.split(',');assert variants[0]=='original'
    op=ScalarOp(args.op);path=HERE/'results'/args.output;assert not path.exists()
    emit(dict(type='environment',device=ACC.get_device_name(),args=vars(args),hashes=hashes(),
              scope='B-only memory and non-timed public dtype / finite update validation'),path)
    for T,H in ((2048,4),(8192,16)):
        case=dict(B=1,T=T,H=H,D=128);xs,dys=op.inputs(case)
        for policy in ('recompute','all'):
            fw=op.forward(xs,policy)
            ref=tuple(x.cpu().double() for x in op.backward(xs,dys,fw,'original'))
            ACC.synchronize()
            for repeat in range(3):
                for variant in variants if repeat%2==0 else variants[::-1]:
                    warm=op.backward(xs,dys,fw,variant);ACC.synchronize();del warm;gc.collect()
                    before=ACC.memory_allocated();ACC.reset_peak_memory_stats()
                    out=op.backward(xs,dys,fw,variant);ACC.synchronize()
                    peak=ACC.max_memory_allocated()-before
                    m=metrics(ref,out,op.names);ok=check(m)
                    emit(dict(type='memory',case=case,policy=policy,variant=variant,repetition=repeat,
                              boundary='warmed complete B incremental allocated peak including returned gradients',
                              backward_peak_extra_bytes=peak,saved_auxiliary_bytes=saved_bytes(fw,xs),
                              gradient_bytes=sum(x.nbytes for x in out),additional_resident_workspace_bytes=0,
                              passed=ok,metrics=m),path)
                    assert ok,(case,policy,variant,m)
                    del out
            if T==2048:
                def public(v):return tuple(d.to(x.dtype) for d,x in zip(op.backward(xs,dys,fw,v),xs))
                reference=public('original')
                for v in variants:
                    actual=public(v);m=metrics(reference,actual,op.names);assert check(m)
                    graph,held=capture(lambda v=v:public(v))
                    other,ody=op.inputs(dict(case,seed=991));ofw=op.forward(other,policy)
                    op.backward(other,ody,ofw,v)
                    graph.replay();ACC.synchronize()
                    replay=all(check(metrics(reference,z,op.names)) for z in held)
                    dtype_ok=all(d.dtype==x.dtype for d,x in zip(actual,xs))
                    assert replay and dtype_ok
                    emit(dict(type='public_dtype',case=case,policy=policy,variant=v,passed=True,
                              timed=False,replay='all captured outputs after other-input interference',
                              dtypes=[str(x.dtype) for x in actual],metrics=m),path)
                    del graph,held,actual,other,ody,ofw
                del reference
            del fw,ref
        del xs,dys;gc.collect();ACC.empty_cache()
    case=dict(B=1,T=256,H=2,D=128);xs,dys=op.inputs(case)
    target=op.oracle(tuple(x.detach().cpu().double() for x in xs))
    target=tuple((x*.5).float().to(DEV) for x in target);dtypes=[x.dtype for x in xs]
    initial=[x.detach().float().clone() for x in xs]
    for policy in ('recompute','all'):
        baseline=None
        for variant in variants:
            master=[x.clone() for x in initial];history=[]
            for step in range(5):
                cur=tuple(x.to(dt) for x,dt in zip(master,dtypes));fw=op.forward(cur,policy)
                out=fw[2];gradout=tuple((2*(o-t)/o.numel()*256*2).to(o.dtype) for o,t in zip(out,target))
                grad=op.backward(cur,gradout,fw,variant)
                loss=sum(((o.float()-t.float())**2).mean().item() for o,t in zip(out,target))
                master=[x-.01*d.float() for x,d in zip(master,grad)];history.append(loss)
            if baseline is None:baseline=tuple(x.clone() for x in master)
            m=metrics(baseline,master,op.names);ok=check(m)
            emit(dict(type='finite_updates',variant=variant,policy=policy,steps=5,learning_rate=.01,
                      loss=history,metrics=m,passed=ok,scope='operator-level consistency, not model convergence'),path)
            assert ok,(policy,variant,m)
    emit(dict(type='dependencies',**dependencies()),path)
    emit(dict(type='complete',passed=True),path)

if __name__=='__main__':
    with ACC.stream(ACC.Stream()):main()
