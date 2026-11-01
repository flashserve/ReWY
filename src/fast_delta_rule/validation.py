"""Public-autograd acceptance against an independent CPU FP64 recurrence."""
import json
import torch
from .runtime import operator,use_backend,stats,check_baseline

def inputs(op,device,length=64,heads=2,seed=42,batch=1):
    torch.manual_seed(seed)
    shape=(batch,length,heads,128)
    q=torch.nn.functional.normalize(torch.randn(shape,device=device),dim=-1).to(torch.bfloat16)
    k=torch.nn.functional.normalize(torch.randn(shape,device=device),dim=-1).to(torch.bfloat16)
    v=torch.randn(shape,device=device,dtype=torch.bfloat16)
    xs=dict(q=q,k=k,v=v)
    if op!='delta':
        gshape=shape[:-1] if op=='gdn' else shape
        xs['g']=-torch.rand(gshape,device=device,dtype=torch.float32)*.05
    if op=='gdn2':
        xs['b']=torch.rand(shape,device=device,dtype=torch.bfloat16)*.8+.1
        xs['w']=torch.rand(shape,device=device,dtype=torch.bfloat16)*.8+.1
    else: xs['beta']=torch.rand(shape[:-1],device=device,dtype=torch.bfloat16)*.8+.1
    xs['initial_state']=torch.randn((batch,heads,128,128),device=device,dtype=torch.float32)*.05
    return {n:x.detach().requires_grad_() for n,x in xs.items()}

def recurrence(op,xs):
    q,k,v=(xs[n] for n in ('q','k','v'))
    state=xs['initial_state'];ys=[]
    for t in range(k.shape[1]):
        if op!='delta':
            g=xs['g'][:,t]
            if op=='gdn': g=g[...,None]
            state=state*g.exp()[...,None]
        kt=k[:,t]
        if op=='gdn2':
            residual=xs['w'][:,t]*v[:,t]-torch.einsum('bhk,bhkv->bhv',xs['b'][:,t]*kt,state)
        else:
            residual=xs['beta'][:,t,...,None]*(v[:,t]-torch.einsum('bhk,bhkv->bhv',kt,state))
        state=state+kt[...,None]*residual[...,None,:]
        ys.append(torch.einsum('bhk,bhkv->bhv',q[:,t]*k.shape[-1]**-.5,state))
    return torch.stack(ys,1),state

def error(ref,got,limit):
    ref=ref.detach().cpu().double();got=got.detach().cpu().double()
    absolute=(got-ref).abs().max().item()
    relative=(got-ref).norm().item()/max(ref.norm().item(),1e-30)
    passed=bool(torch.isfinite(got).all()) and (absolute<=1e-6 if ref.abs().max()<1e-6 else relative<=limit)
    return {'relative_l2':relative,'max_abs':absolute,'limit':limit,'passed':passed}

def verify(device='cuda',ops=('kda','gdn','gdn2','delta')):
    if device=='npu':
        import torch_npu  # Registers the device even when backend autoload is disabled.
    commit=check_baseline();rows=[]
    stats(reset=True)
    for op in ops:
        xs=inputs(op,device)
        cpu={n:x.detach().cpu().double().requires_grad_() for n,x in xs.items()}
        assert all(x.device.type == 'cpu' and x.dtype == torch.float64 for x in cpu.values())
        do=torch.randn_like(xs['v']);dh=torch.randn_like(xs['initial_state'])
        expected=recurrence(op,cpu)
        refgrads=torch.autograd.grad(expected,tuple(cpu.values()),(do.cpu().double(),dh.cpu().double()))
        policies=('recompute','all') if op in ('kda','gdn2') else ('recompute',)
        for policy in policies:
            kw=dict(xs,output_final_state=True)
            if op in ('kda','gdn2'): kw['disable_recompute']=policy=='all'
            for backend in ('original','optimized'):
                with use_backend(backend,ops=(op,)):
                    out=operator(op)(**kw)
                # Run after leaving the context: backward must remain bound.
                grads=torch.autograd.grad(out,tuple(xs.values()),(do,dh))
                metrics={name:error(ref,actual,.02 if name in ('g','beta','b','w') else .008)
                         for name,ref,actual in zip(xs,refgrads,grads)}
                metrics['output']=error(expected[0],out[0],.008)
                metrics['final_state']=error(expected[1],out[1],.008)
                row={'op':op,'backend':backend,'policy':policy,'metrics':metrics,'passed':all(m['passed'] for m in metrics.values())}
                rows.append(row)
                print(json.dumps({'op':op,'backend':backend,'policy':policy,'passed':row['passed']}),flush=True)
    return {'passed':all(r['passed'] for r in rows),'device':device,'baseline':commit,'rows':rows,'dispatch':stats(),
            'boundary':'public forward and complete autograd backward; no performance measurement'}
