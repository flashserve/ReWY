"""Paired public-interface timing, distinct from historical low-level results."""
import statistics
import time
import torch
from .runtime import check_baseline,operator,use_backend,stats
from .validation import inputs,error

def benchmark(device,op,length,heads,rounds,boundary,policy,batch=1):
    if device=='npu':
        import torch_npu
    if rounds<1: raise ValueError('rounds must be positive')
    if policy=='all' and op not in ('kda','gdn2'):
        raise ValueError('The upstream public GDN/DeltaNet API has no all-save option; do not silently change its save policy.')
    commit=check_baseline()
    xs=inputs(op,device,length,heads,batch=batch)
    kw=dict(xs,output_final_state=True)
    if op in ('kda','gdn2'):kw['disable_recompute']=policy=='all'
    do=torch.randn_like(xs['v']);dh=torch.randn_like(xs['initial_state'])
    sync=getattr(torch,device).synchronize
    def forward(backend):
        with use_backend(backend,ops=(op,)):return operator(op)(**kw)
    def backward(outputs):return torch.autograd.grad(outputs,tuple(xs.values()),(do,dh))
    reference=backward(forward('original'));candidate=backward(forward('optimized'))
    checks={n:error(r,c,.02 if n in ('beta','b','w','g') else .008) for n,r,c in zip(xs,reference,candidate)}
    if not all(v['passed'] for v in checks.values()): raise RuntimeError(f'Gradient check failed: {checks}')
    def sample(backend):
        out=forward(backend) if boundary=='B' else None
        sync();start=time.perf_counter()
        if boundary=='FB':out=forward(backend)
        grads=backward(out)
        sync();elapsed=(time.perf_counter()-start)*1e3
        del grads,out
        return elapsed
    start=time.perf_counter();warmups=0
    while warmups<6 or time.perf_counter()-start<2:
        for backend in ('original','optimized'):sample(backend)
        warmups+=1
    samples={'original':[],'optimized':[]};orders=[]
    stats(reset=True)
    for r in range(rounds):
        order=['original','optimized'] if r%2==0 else ['optimized','original']
        order=order+order[::-1];orders.append(order)
        for backend in order:samples[backend].append(sample(backend))
    medians={n:statistics.median(v) for n,v in samples.items()}
    if any('fallback' in key for key in stats()):
        raise RuntimeError('Benchmark encountered fallback; no speedup is reported.')
    return dict(op=op,device=device,baseline=commit,shape=list(xs['q'].shape),policy=policy,boundary=boundary,
                timing='public autograd, synchronized eager wall-clock; allocation and gradient conversion included',
                warmups=warmups,rounds=rounds,orders=orders,samples_ms=samples,median_ms=medians,
                speedup=medians['original']/medians['optimized'],correctness=checks,dispatch=stats())
