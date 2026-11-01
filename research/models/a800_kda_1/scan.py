"""Fixed-model KDA shape exploration; one resident model, ABBA blocks."""
import os
import argparse, contextlib, copy, functools, gc, importlib, json, statistics, sys, time
from pathlib import Path
BASE=Path(__file__).resolve().parent.parent/'a800_shared'
sys.path.insert(0,str(BASE))
import integration as impl
from models import make_model,loss_for,TRACE
import torch
from graph_training import GraphedTrainingStep
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
ROOT=Path(os.environ.get('FDR_MODEL_OUTPUT',str(Path(__file__).resolve().parent/'results'))).resolve()
ROOT.mkdir(parents=True,exist_ok=True)

def emit(kind,**kw):
    row=dict(type=kind,time=time.time(),**kw)
    with (ROOT/'scan.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

def snapshot(m,o,grads=False):
    return dict(parameters={n:p.detach().cpu() for n,p in m.named_parameters()},
        optimizer={n:{k:v.detach().cpu() for k,v in o.state[p].items()} for n,p in m.named_parameters()},
        gradients={n:p.grad.detach().cpu() for n,p in m.named_parameters()} if grads else {})

def restore(m,o,s):
    with torch.no_grad():
        for n,p in m.named_parameters():
            p.copy_(s['parameters'][n])
            for k,v in o.state[p].items():v.copy_(s['optimizer'][n][k])

def exact_check(m,o,s):
    checks=0
    for n,p in m.named_parameters():
        assert torch.equal(p.detach().cpu(),s['parameters'][n]),n
        assert torch.equal(p.grad.detach().cpu(),s['gradients'][n]),n+'.grad'
        checks+=2
        for k,v in o.state[p].items():
            assert torch.equal(v.detach().cpu(),s['optimizer'][n][k]),n+'.'+k
            checks+=1
    return checks

def compare_cpu(m,ref,tag,grad=False,enforce=True):
    metrics={};total=error=0.
    for n,p in m.named_parameters():
        a=ref[n].reshape(-1);b=(p.grad if grad else p).detach().reshape(-1)
        norm=diff=maximum=absolute=0.
        for i in range(0,a.numel(),2**20):
            x=a[i:i+2**20].to('cuda',dtype=torch.float64);y=b[i:i+2**20].double();d=y-x
            norm+=x.square().sum().item();diff+=d.square().sum().item()
            maximum=max(maximum,x.abs().max().item());absolute=max(absolute,d.abs().max().item())
        rel=(diff/max(norm,1e-60))**.5
        limit=.02 if any(k in n for k in ('A_log','dt_bias','f_proj','b_proj','w_proj')) else .008
        finite=bool(torch.isfinite(torch.tensor([norm,diff,maximum,absolute],dtype=torch.float64)).all())
        assert finite,(tag,n)
        metrics[n]=dict(relative_l2=rel,max_abs=absolute,finite=finite,outside_operator_budget=absolute>1e-6 if maximum<1e-6 else rel>limit)
        total+=norm;error+=diff
    rel=(error/max(total,1e-60))**.5
    (ROOT/(tag+'.json')).write_text(json.dumps(metrics,indent=2))
    emit('network_check',tag=tag,global_relative_l2=rel,enforced=enforce,passed=rel<=.008 if enforce else None,
         outside_operator_budget=sum(x['outside_operator_budget'] for x in metrics.values()))
    if enforce:assert rel<=.008,(tag,rel)

@contextlib.contextmanager
def core_markers():
    cls=importlib.import_module('fla.ops.kda.chunk').ChunkKDAFunction
    raw=cls.backward;pairs=[]
    @functools.wraps(raw)
    def wrapped(*a,**kw):
        x,y=[torch.cuda.Event(enable_timing=True,external=True) for _ in range(2)]
        pairs.append((x,y));x.record();out=raw(*a,**kw);y.record();return out
    cls.backward=staticmethod(wrapped)
    try:yield pairs
    finally:cls.backward=staticmethod(raw)

def eager(m,o,t,v,update=True):
    with impl.implementation('kda',v):
        o.zero_grad(set_to_none=True);loss=loss_for(m,'kda',t,t);loss.backward()
        if update:o.step()
    return loss

def block(v,idx,args,refs):
    impl.AUDITS.clear();impl.AUDIT_LOCAL=False
    if 'template' not in refs:
        template,info=make_model('kda');refs['template']=template;refs['info']=info
    m=copy.deepcopy(refs['template']);info=refs['info'];m.cuda()
    o=torch.optim.AdamW(m.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=.1,eps=1e-8,fused=True,capturable=True)
    torch.manual_seed(991);tokens=torch.randint(info['vocab_size'],(args.batch,args.length),device='cuda')
    original_tokens=tokens.clone();changed=torch.randint(info['vocab_size'],tokens.shape,device='cuda')
    emit('model',block=idx,variant=v,batch=args.batch,length=args.length,**info)
    impl.AUDIT_LOCAL=idx==1
    loss=eager(m,o,tokens,v,False)
    impl.AUDIT_LOCAL=False
    initial_loss=loss.item()
    if idx==0:
        refs['initial_loss']=initial_loss
        refs['gradients']={n:p.grad.detach().cpu() for n,p in m.named_parameters()}
    if idx==1:
        assert initial_loss==refs['initial_loss']
        assert len(impl.AUDITS)==24 and all(r['passed'] for r in impl.AUDITS)
        (ROOT/'local_vjp.json').write_text(json.dumps(impl.AUDITS,indent=2))
        emit('local_vjp',passed=True,layers=24,trace=dict(TRACE))
        compare_cpu(m,refs.pop('gradients'),'initial_gradients',grad=True,enforce=False)
    o.step()
    for _ in range(5):eager(m,o,tokens,v)
    torch.cuda.synchronize()
    before=next(iter(o.state.values()))['step'].item()
    graph=GraphedTrainingStep(m,o,lambda:loss_for(m,'kda',tokens,tokens),lambda:impl.implementation('kda',v))
    assert next(iter(o.state.values()))['step'].item()==before==6
    tokens.copy_(changed)
    if idx<2:
        saved=snapshot(m,o)
        expected_loss=eager(m,o,tokens,v).item();expected=snapshot(m,o,True)
        restore(m,o,saved);del saved
        got=graph.replay();torch.cuda.synchronize()
        assert expected_loss==got.item()
        checks=exact_check(m,o,expected);del expected
        emit('graph_equivalence',variant=v,changed_input=True,bitwise_equal=True,checks=checks)
    else:graph.replay()
    tokens.copy_(original_tokens)
    for _ in range(9):graph.replay()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    emit('ready',variant=v,block=idx,allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
    for sample in range(args.samples):
        torch.cuda.synchronize();start=time.perf_counter();loss=graph.replay();submitted=(time.perf_counter()-start)*1000
        torch.cuda.synchronize();wall=(time.perf_counter()-start)*1000
        phases=graph.phase_ms();value=loss.item();assert torch.isfinite(loss).item()
        emit('sample',variant=v,block=idx,sample=sample,wall_ms=wall,submit_ms=submitted,phase_ms=phases,loss=value,
             allocated_peak=torch.cuda.max_memory_allocated(),reserved_peak=torch.cuda.max_memory_reserved())
    assert all(s['step'].item()==16+args.samples for s in o.state.values())
    with torch.no_grad():final_loss=loss_for(m,'kda',tokens,tokens).item()
    if idx==0:
        refs['parameters']={n:p.detach().cpu() for n,p in m.named_parameters()}
        refs['final_loss']=final_loss
    if idx==1:
        compare_cpu(m,refs.pop('parameters'),'final_parameters')
        rel=abs(final_loss-refs['final_loss'])/max(abs(refs['final_loss']),1e-30)
        assert rel<=.008
        emit('final_loss_check',passed=True,relative_error=rel)
    emit('backend',variant=v,block=idx,**impl.backend_snapshot())
    del graph; o.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
    if idx<2:
        with core_markers() as pairs:
            diagnostic=GraphedTrainingStep(m,o,lambda:loss_for(m,'kda',tokens,tokens),lambda:impl.implementation('kda',v))
        assert len(pairs)==24,len(pairs)
        for _ in range(3):diagnostic.replay()
        for sample in range(10):
            torch.cuda.synchronize();start=time.perf_counter();diagnostic.replay();torch.cuda.synchronize()
            emit('core_diagnostic',variant=v,sample=sample,wall_ms=(time.perf_counter()-start)*1000,
                 phase_ms=diagnostic.phase_ms(),core_b_ms=sum(a.elapsed_time(b) for a,b in pairs))
        del diagnostic,pairs
    o.zero_grad(set_to_none=True)
    del loss,m,o,tokens,original_tokens,changed
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    emit('block_complete',variant=v,block=idx,allocated=torch.cuda.memory_allocated())

def main(args):
    refs={}
    for idx,v in enumerate(('original','optimized','optimized','original')):block(v,idx,args,refs)
    emit('complete',passed=True,batch=args.batch,length=args.length,samples_per_variant=args.samples*2,
         boundary='full updating Graph STEP; B is interval within STEP; core diagnostic separately marked',
         policy='native R; no layer checkpoint; frozen model and kernel; ABBA fresh-model blocks')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,required=True);p.add_argument('--length',type=int,required=True);p.add_argument('--samples',type=int,default=30);args=p.parse_args()
    try:
        with torch.cuda.stream(torch.cuda.Stream()):main(args)
    except Exception as exc:emit('failure',error=repr(exc));raise
