"""Extra head sizes and FP16 checks for a retained uniform GPU schedule."""
import argparse,gc
from common import *
from scalar_ops import ScalarOp

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--op',required=True)
    ap.add_argument('--variants',required=True);ap.add_argument('--output',required=True)
    a=ap.parse_args();path=HERE/'results'/a.output
    if path.exists():raise FileExistsError(path)
    emit(dict(type='environment',device=DEV,args=vars(a),hashes=hashes()),path)
    op=ScalarOp(a.op);failed=[];unsupported=[]
    for case_id,case in enumerate([dict(B=1,T=97,H=2,D=80,V=112),
                                   dict(B=1,T=65,H=2,D=192,V=160),
                                   dict(B=1,T=129,H=2,D=96,V=80,dtype=torch.float16)]):
        xs,dys=op.inputs(case)
        cpu=tuple(x.detach().cpu().double().requires_grad_() for x in xs)
        out=op.oracle(cpu);ref=torch.autograd.grad(out,cpu,tuple(x.cpu().double() for x in dys))
        for policy in ('recompute','all'):
            fw=op.forward(xs,policy)
            for variant in a.variants.split(','):
                try:
                    grads=op.backward(xs,dys,fw,variant)
                except triton.runtime.errors.OutOfResources as error:
                    record=dict(type='unsupported_resources',case_id=case_id,
                                case={k:str(v) if k=='dtype' else v for k,v in case.items()},
                                variant=variant,policy=policy,supported=False,
                                error_type=type(error).__name__,reason=str(error))
                    emit(record,path);unsupported.append((case_id,policy,variant))
                    failed.append((case_id,policy,variant))
                    continue
                m=metrics(ref,grads,op.names);m.update(metrics(out,fw[2],['output','final_state']))
                emit(dict(type='correctness',case_id=case_id,case={k:str(v) if k=='dtype' else v for k,v in case.items()},
                          variant=variant,policy=policy,passed=check(m),metrics=m),path)
                if not check(m):failed.append((case_id,policy,variant))
        del xs,dys,cpu,out,ref,fw,grads;gc.collect();ACC.empty_cache()
    emit(dict(type='dependencies',**dependencies()),path)
    emit(dict(type='complete',passed=not failed,failures=failed,unsupported=unsupported,
              required_full_coverage=a.variants.split(','),
              scope='Every selected variant must support and pass every extra case.'),path)
    if failed:raise RuntimeError(str(failed))
if __name__=='__main__':
    with ACC.stream(ACC.Stream()):main()
