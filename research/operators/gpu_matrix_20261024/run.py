"""Reproducible GPU correctness and backward-only ablation harness."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import time
import traceback
import numpy as np
import torch
import triton
from op import e, implementation, VARIANTS, ROOT

HERE = Path(__file__).resolve().parent
CASES = [dict(B=1,T=512,H=4,D=128), dict(B=1,T=2048,H=4,D=128),
         dict(B=1,T=8192,H=4,D=128), dict(B=1,T=2048,H=16,D=128),
         dict(B=2,T=2048,H=16,D=128), dict(B=1,T=8192,H=16,D=128)]
NAMES = e.GRAD_NAMES


def emit(record, path):
    record['time'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record) + '\n')
    short = {k:v for k,v in record.items() if k not in ('metrics','samples_us','hashes','telemetry')}
    print(json.dumps(short), flush=True)


def hashes():
    files = list(HERE.glob('*.py')) + list((ROOT/'kernels').glob('*.py')) + [ROOT/'run_cuda.py']
    files += list((ROOT/'vendor/fla/fla/ops/kda').glob('*.py'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def metrics(ref, actual, names=NAMES):
    return {n:e.error(a,b,.02 if n in ('dg','dbeta') else .008) for n,a,b in zip(names,ref,actual)}


def passed(ms):
    return all(m['pass'] for m in ms.values())


def flags_info(obj):
    if obj.flags is None:
        return dict(calls=obj.calls, api_fallbacks=obj.fallback_calls)
    f = obj.flags.cpu()
    return dict(calls=obj.calls, api_fallbacks=obj.fallback_calls,
                tiles=f.numel(), pairwise_tiles=int(f.sum()),
                valid_flags=bool(((f==0)|(f==1)).all()),
                registers=obj.kernel.n_regs, spills=obj.kernel.n_spills,
                shared_memory=obj.kernel.metadata.shared)


def forward(xs, policy, **kw):
    return e.chunk_kda(*xs[:5], initial_state=xs[5], output_final_state=True,
                       disable_recompute=policy=='all', **kw)


def correctness_cases():
    cases = [dict(name='native64',T=64,D=128),dict(name='native128',T=128,D=128),
             dict(name='tail129',T=129,D=128),dict(name='batch_gva_tail',B=2,T=127,D=64,HV=4,V=96),
             dict(name='state_v_first',T=193,D=128,state_v_first=True),
             dict(name='strong_gate',T=128,D=128,gate_div=.5),
             dict(name='weak_gate',T=129,D=128,gate_div=100),
             dict(name='zero_beta',T=65,D=128,beta_mode='zero'),
             dict(name='tiny_beta',T=65,D=128,beta_mode='tiny'),
             dict(name='beta_over_one',T=65,D=128,beta_mode='over_one'),
             dict(name='zero_gate',T=65,D=128,gate_mode='zero'),
             dict(name='mixed_tiles',T=128,D=128,gate_mode='mixed'),
             dict(name='nonmonotonic_gate',T=128,D=128,gate_mode='mixed_sign'),
             dict(name='fp16_chunk32',T=65,D=64,dtype='fp16',chunk_size=32),
             dict(name='single_token',T=1,D=128),
             dict(name='packed_tails_gva',T=226,D=64,HV=4,V=96,lengths=[1,31,32,33,64,65]),
             dict(name='nonpower_dims',T=97,D=96,V=80),
             dict(name='seed7',T=257,D=128,seed=7),
             dict(name='seed123',T=257,D=128,seed=123),
             dict(name='seed997',T=257,D=128,seed=997),
             dict(name='long_strong',T=1024,D=64,seed=27,gate_div=1),
             dict(name='long_weak',T=2048,D=64,seed=37,gate_div=100),
             dict(name='no_initial_no_final',T=97,D=64,initial=False,final=False),
             dict(name='no_initial_final',T=97,D=64,initial=False,final=True),
             dict(name='initial_no_final',T=97,D=64,initial=True,final=False)]
    return cases


def make_case(case):
    settings = dict(B=1,H=2,D=128,T=128)
    settings.update({k:v for k,v in case.items() if k in ('B','T','H','D','HV','V','seed','gate_div','state_v_first')})
    settings['dtype'] = torch.float16 if case.get('dtype')=='fp16' else torch.bfloat16
    xs,dys = e.inputs(**settings)
    with torch.no_grad():
        if case.get('beta_mode')=='zero': xs[4][:,::3]=0
        if case.get('beta_mode')=='tiny': xs[4][:,::3]*=1.e-8
        if case.get('beta_mode')=='over_one': xs[4].mul_(2)
        if case.get('gate_mode')=='zero': xs[3].zero_()
        if case.get('gate_mode')=='mixed': xs[3][:,:64,0,:64].mul_(100)
        if case.get('gate_mode')=='mixed_sign': xs[3][:,::3].mul_(-1)
    lengths = case.get('lengths')
    offsets = [0]
    if lengths:
        for size in lengths: offsets.append(offsets[-1]+size)
        assert offsets[-1] == xs[0].shape[1]
        xs=(*xs[:-1],xs[-1].detach().repeat(len(lengths),1,1,1).requires_grad_())
        dys=(dys[0],dys[1].repeat(len(lengths),1,1,1))
    cpu=tuple(x.detach().cpu().double().requires_grad_() for x in xs)
    if not case.get('initial',True): cpu=(*cpu[:-1],torch.zeros_like(cpu[-1]))
    svf=case.get('state_v_first',False)
    if lengths:
        pieces=[e.oracle((*[z[:,a:b] for z in cpu[:-1]],cpu[-1][j:j+1]),svf)
                for j,(a,b) in enumerate(zip(offsets[:-1],offsets[1:]))]
        ref=(torch.cat([p[0] for p in pieces],1),torch.cat([p[1] for p in pieces],0))
    else:
        ref=e.oracle(cpu,svf)
    kw=dict(initial_state=xs[-1] if case.get('initial',True) else None,
            output_final_state=case.get('final',True),chunk_size=case.get('chunk_size',64),state_v_first=svf)
    if lengths: kw['cu_seqlens']=torch.tensor(offsets,device='cuda',dtype=torch.int64)
    if not case.get('final',True):
        ref,dys=ref[:1],dys[:1]
    cpu_args=cpu if case.get('initial',True) else cpu[:5]
    refgrad=torch.autograd.grad(ref,cpu_args,tuple(x.cpu().double() for x in dys))
    return xs,dys,kw,ref,refgrad


def correctness(args,path):
    cases=correctness_cases()
    if args.case is not None: cases=[cases[args.case]]
    failures=[]
    for case in cases:
        print('CASE '+case['name'],flush=True)
        xs,dys,kw,ref,refgrad=make_case(case)
        for policy in args.policies.split(','):
            forward_bytes=None
            for variant in args.variants.split(','):
                try:
                    with implementation(variant,args.bk,args.warps,not args.fp32) as obj:
                        out=e.chunk_kda(*xs[:5],**kw,disable_recompute=policy=='all')
                        if not case.get('final',True): out=out[:1]
                        grads=torch.autograd.grad(out,xs if case.get('initial',True) else xs[:5],dys)
                        ms=metrics(refgrad,grads)
                        ms.update(metrics(ref,out,('output','final_state')))
                        exact=True if forward_bytes is None else all(torch.equal(a,b) for a,b in zip(forward_bytes,out))
                        if forward_bytes is None: forward_bytes=tuple(x.detach().clone() for x in out)
                        dispatch=flags_info(obj)
                    ok=passed(ms) and exact and dispatch.get('valid_flags',True)
                    emit(dict(type='correctness',case=case,policy=policy,variant=variant,passed=ok,
                              metrics=ms,forward_bitwise_equal=exact,dispatch=dispatch),path)
                except Exception as exc:
                    traceback.print_exc()
                    ok=False
                    emit(dict(type='correctness',case=case,policy=policy,variant=variant,passed=False,error=repr(exc)),path)
                if not ok: failures.append((case['name'],policy,variant))
        del xs,dys,ref,refgrad
        gc.collect()
        torch.cuda.empty_cache()
    if failures: raise RuntimeError(f'correctness failures: {failures}')


def capture(variant, xs, dys, out, args, ref):
    held=[]
    flagrefs=[]
    with implementation(variant,args.bk,args.warps,not args.fp32) as obj:
        for _ in range(3):
            grad=torch.autograd.grad(out,xs,dys,retain_graph=True)
        torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=torch.cuda.current_stream()):
            for _ in range(args.unroll):
                held.append(torch.autograd.grad(out,xs,dys,retain_graph=True))
                if obj.flags is not None: flagrefs.append(obj.flags)
        graph.replay()
        torch.cuda.synchronize()
        for g in held:
            if not passed(metrics(ref,g)): raise RuntimeError('initial graph replay precision: '+variant)
        f=flags_info(obj)
        if not f.get('valid_flags',True): raise RuntimeError('uninitialized dispatch flags')
    return graph,held,flagrefs,f


def pair_timing(graphs, rounds, unroll):
    names=list(graphs)
    start=time.monotonic()
    while time.monotonic()-start<2:
        for name in names: graphs[name].replay()
        torch.cuda.synchronize()
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(8):
        for name in names: graphs[name].replay()
    b.record();torch.cuda.synchronize()
    one=a.elapsed_time(b)/(8*len(names))
    replays=max(1,min(2048,int(60/max(one,.001))))
    samples={n:[] for n in names}
    orders=[]
    for r in range(rounds):
        first=names[r%len(names):]+names[:r%len(names)]
        if r%2: first=first[::-1]
        order=first+first[::-1]
        orders.append(order)
        ev=[]
        for name in order:
            a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(replays): graphs[name].replay()
            b.record();ev.append((name,a,b))
        torch.cuda.synchronize()
        for name,a,b in ev: samples[name].append(a.elapsed_time(b)*1000/(replays*unroll))
    return samples,replays,orders


def bench(args,path):
    indices=range(len(CASES)) if args.case is None else [args.case]
    for index in indices:
        case=CASES[index]
        for policy in args.policies.split(','):
            print(f'BENCH case={index} policy={policy}',flush=True)
            xs,dys=e.inputs(**case)
            # One public, unchanged forward shared by every backward candidate.
            out=forward(xs,policy)
            with implementation('original'):
                ref=tuple(x.detach().clone() for x in torch.autograd.grad(out,xs,dys,retain_graph=True))
            names=args.variants.split(',')
            graphs,held,flagrefs,dispatch={},{},{},{}
            for name in names:
                with implementation(name,args.bk,args.warps,not args.fp32):
                    grad=torch.autograd.grad(out,xs,dys,retain_graph=True)
                ms=metrics(ref,grad)
                emit(dict(type='benchmark_correctness',case_id=index,case=case,policy=policy,variant=name,
                          metrics=ms,passed=passed(ms)),path)
                if not passed(ms): raise RuntimeError('gradient check before timing: '+name)
                graphs[name],held[name],flagrefs[name],dispatch[name]=capture(name,xs,dys,out,args,ref)
            samples,replays,orders=pair_timing(graphs,args.rounds,args.unroll)
            # Test old captures after unrelated invocations on different input storage.
            other,otherdy=e.inputs(**dict(case,seed=991))
            for name in names:
                with implementation(name,args.bk,args.warps,not args.fp32):
                    torch.autograd.grad(forward(other,policy),other,otherdy)
            post={}
            for name in names:
                graphs[name].replay();torch.cuda.synchronize()
                ms=metrics(ref,held[name][-1])
                if not all(passed(metrics(ref,g)) for g in held[name]):
                    raise RuntimeError('post timing replay precision: '+name)
                post[name]=ms
            median={n:statistics.median(v) for n,v in samples.items()}
            speed={n:median['original']/v for n,v in median.items()} if 'original' in median else {}
            emit(dict(type='benchmark',case_id=index,case=case,policy=policy,boundary='complete_backward',
                      median_us=median,speedup_vs_original=speed,samples_us=samples,rounds=args.rounds,
                      unroll=args.unroll,replays=replays,orders=orders,dispatch=dispatch,
                      metrics=post,passed=True,telemetry=e.telemetry(),
                      method='CUDA graph; >=2s warmup; rotated palindromic order; 2 observations/round/variant'),path)
            del graphs,held,flagrefs,out,ref,xs,dys,grad,other,otherdy
            gc.collect();torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--mode',choices=('correctness','bench'),required=True)
    ap.add_argument('--variants',default=','.join(VARIANTS))
    ap.add_argument('--policies',default='recompute,all')
    ap.add_argument('--bk',type=int,default=32)
    ap.add_argument('--warps',type=int,default=4)
    ap.add_argument('--fp32',action='store_true')
    ap.add_argument('--case',type=int)
    ap.add_argument('--rounds',type=int,default=10)
    ap.add_argument('--unroll',type=int,default=2)
    ap.add_argument('--output',required=True)
    args=ap.parse_args()
    path=HERE/'results'/args.output
    path.parent.mkdir(exist_ok=True)
    if path.exists(): raise FileExistsError('Use a fresh output name: '+str(path))
    start=hashes()
    emit(dict(type='environment',args=vars(args),torch=torch.__version__,triton=triton.__version__,
              device=str(torch.cuda.get_device_properties(0)),hashes=start,telemetry=e.telemetry()),path)
    with torch.cuda.stream(torch.cuda.Stream()):
        if args.mode=='correctness': correctness(args,path)
        else: bench(args,path)
    if hashes()!=start: raise RuntimeError('source changed during run')
    emit(dict(type='complete',passed=True,source_unchanged=True),path)


if __name__=='__main__':
    main()
