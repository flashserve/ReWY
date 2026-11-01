import argparse,gc,time
from common import *
from scalar_ops import ScalarOp

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--op',required=True);ap.add_argument('--mode',choices=['correctness','bench'],required=True)
    ap.add_argument('--case',type=int);ap.add_argument('--rounds',type=int,default=10);ap.add_argument('--policies',default='recompute,all');ap.add_argument('--variants');ap.add_argument('--boundaries',default='B,FB');ap.add_argument('--output',required=True)
    a=ap.parse_args();path=HERE/'results'/a.output;path.parent.mkdir(exist_ok=True)
    if path.exists():raise FileExistsError(path)
    if a.op=='gdn2':
        raise ValueError('Use the vector paper entry for GDN2')
        op=GDN2()
    elif a.op=='kda':
        raise ValueError('Use the vector paper entry for KDA')
        op=KDA()
    else:op=ScalarOp(a.op)
    variants=a.variants.split(',') if a.variants else op.variants
    if not set(variants) <= {'original', 'liveness', 'native_v16s3'}:raise ValueError('Variant is outside the published paper artifact')
    emit(dict(type='environment',device=DEV,torch=torch.__version__,triton=triton.__version__,args=vars(a),tuning={k:v for k,v in os.environ.items() if k.startswith('GDN_')},hashes=hashes()),path)
    cases=CASES if a.mode=='bench' else [dict(B=1,T=64,H=2,D=64),dict(B=1,T=129,H=2,D=128),dict(B=2,T=127,H=2,HV=4,D=64,V=96),dict(B=1,T=193,H=2,D=128,state_v_first=a.op=='gdn'),dict(B=1,T=65,H=2,D=64,gate_div=1,zero_beta=True),dict(B=1,T=129,H=2,D=64,gate_div=100,tiny_beta=True),dict(B=1,T=97,H=2,D=64,positive_gate=True),dict(B=1,T=2048,H=2,D=64)]
    if a.op in ('delta','gdn2'):
        cases=[dict(c) for c in cases]
        for c in cases:c.pop('HV',None)
        if a.mode=='correctness' and a.op=='gdn2':cases[3]['state_v_first']=True
    failures=[]
    for i,case in enumerate(cases):
        if a.case is not None and a.case!=i:continue
        print('START',a.op,a.mode,i,case,flush=True)
        xs,dys=op.inputs(case);svf=case.get('state_v_first',False)
        if a.mode=='correctness':
            cpu=tuple(x.detach().cpu().double().requires_grad_() for x in xs)
            out=op.oracle(cpu,svf);ref=torch.autograd.grad(out,cpu,tuple(x.cpu().double() for x in dys))
        for policy in a.policies.split(','):
            fw=op.forward(xs,policy,svf)
            if a.mode=='correctness' and hasattr(op,'gate_diagnostics'):
                emit(dict(type='gate_diagnostics',case_id=i,policy=policy,**op.gate_diagnostics(fw)),path)
            if a.mode=='bench':ref=tuple(x.clone() for x in op.backward(xs,dys,fw,'original',svf))
            for v in variants:
                grads=op.backward(xs,dys,fw,v,svf);m=metrics(ref,grads,op.names)
                if a.mode=='correctness':m.update(metrics(out,fw[2],['output','final_state']))
                emit(dict(type='correctness',case_id=i,case=case,policy=policy,variant=v,passed=check(m),metrics=m),path)
                if not check(m):
                    failures.append((i,policy,v))
                    if a.mode=='bench':raise RuntimeError('correctness failed')
            if a.mode=='bench':
                # Keep only one boundary's graphs alive, so the largest shape
                # remains within the laptop GPU budget. Each boundary is paired.
                for boundary in a.boundaries.split(','):
                    if DEV=='npu' and a.op=='gdn2' and boundary=='FB':
                        # Frozen upstream GDN2 forward contains mandatory host
                        # stream synchronizations, which cannot be captured.
                        funcs={boundary+':'+v:(lambda v=v:op.backward(xs,dys,op.forward(xs,policy,svf),v,svf)) for v in variants}
                        t=eager_timing(funcs,a.rounds)
                        for key,fn in funcs.items():
                            if not check(metrics(ref,fn(),op.names)):raise RuntimeError('eager FB correctness failed '+key)
                        emit(dict(type='benchmark',case_id=i,case=case,policy=policy,boundary=boundary,passed=True,validation='post_timing_full_gradient_comparison',**t),path)
                        del funcs;gc.collect();ACC.empty_cache()
                        continue
                    graphs={};held={}
                    for v in variants:
                        key=boundary+':'+v
                        fn=(lambda v=v:op.backward(xs,dys,fw,v,svf)) if boundary=='B' else (lambda v=v:op.backward(xs,dys,op.forward(xs,policy,svf),v,svf))
                        graphs[key],held[key]=capture(fn)
                    t=timing(graphs,a.rounds)
                    other,ody=op.inputs(dict(case,seed=991));ofw=op.forward(other,policy,svf)
                    for v in variants:op.backward(other,ody,ofw,v,svf)
                    for v,graph in graphs.items():
                        graph.replay();ACC.synchronize()
                        if not all(check(metrics(ref,z,op.names)) for z in held[v]):raise RuntimeError('replay correctness failed '+v)
                    emit(dict(type='benchmark',case_id=i,case=case,policy=policy,boundary=boundary,passed=True,timing_method='device_graph_event',validation='perturbation_and_all_output_replay',**t),path)
                    del other,ody,ofw,graphs,held,graph;gc.collect();ACC.empty_cache()
            del fw
        del xs,dys,ref;gc.collect();ACC.empty_cache()
    emit(dict(type='dependencies',**dependencies()),path)
    emit(dict(type='complete',passed=not failures,failures=failures),path)
    if failures:raise RuntimeError('accuracy failures: '+str(failures))
if __name__=='__main__':
    with ACC.stream(ACC.Stream()):main()
