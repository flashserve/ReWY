"""Full official-size language-model training on one physical Ascend910_9382.

Single model residency, four ABBA blocks, same initialization per block.
STEP wall time includes gradient clearing, forward/loss, backward and AdamW.
B is separately synchronized full-model backward; FB excludes AdamW/zero-grad.
No host input creation, validation, compilation or checkpoint reload is timed.
"""
import argparse, collections, gc, hashlib, importlib, json, os
from pathlib import Path
import statistics, subprocess, time, traceback, zipfile, re, threading, sys
import torch, torch_npu

R=Path(__file__).resolve().parent
SMI_CARD=int(os.environ.get('FDR_NPU_SMI_CARD','7'))
SMI_CHIP=int(os.environ.get('FDR_NPU_SMI_CHIP','1'))

def emit(kind,**data):
    item={'type':kind,'time':time.time(),**data}
    with (OUT/'records.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
    print(json.dumps(item),flush=True)

def health():
    p=subprocess.run(['npu-smi','info'],capture_output=True,text=True,timeout=15)
    return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}

def monitor(stop,errors):
    while not stop.is_set():
        try:
            item=health();item['time']=time.time()
            assert item['returncode']==0,item
            owners=[int(x) for x in re.findall(rf'\|\s*{SMI_CARD}\s+{SMI_CHIP}\s*\|\s*(\d+)\s*\|',item['stdout'])]
            item.update(npu_smi_card=SMI_CARD,npu_smi_chip=SMI_CHIP,visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),owners=owners)
            if any(p!=os.getpid() for p in owners):errors.append(item)
            with (OUT/'telemetry.jsonl').open('a') as f:f.write(json.dumps(item)+'\n')
        except Exception as e:errors.append({'error':repr(e)})
        stop.wait(3)

def snapshot():
    files=[p for p in R.glob('*.py')]
    files+=list((R.parents[2]/'src/fast_delta_rule').rglob('*.py'))
    files+=list(Path(importlib.import_module('fla').__file__).resolve().parent.rglob('*.py'))
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

def build(a):
    if a.family=='kda':
        from fla.models.kda import KDAConfig,KDAForCausalLM
        torch.manual_seed(20261027)
        cfg=KDAConfig(use_cache=False)
        m=KDAForCausalLM(cfg)
        info={'model':'FLA native KDAForCausalLM default','scope':'official_FLA_default','config':cfg.to_dict(),
              'layers':24,'parameters':sum(p.numel() for p in m.parameters())}
    else:
        from native_models import make_model
        m,info=make_model(a)
    assert info['layers'] in (18,24)
    return m.train(),info

def loss(m,x):
    with torch.autocast('npu',dtype=torch.bfloat16):
        return m(input_ids=x,labels=x,use_cache=False).loss

def finite_model(m):
    vals=[]
    for p in m.parameters():
        vals.append(p.isfinite().all())
        if p.grad is not None:vals.append(p.grad.isfinite().all())
    return bool(torch.stack(vals).all().item())

def run(a):
    import adapters as ad
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    assert torch.npu.get_device_name(0)=='Ascend910_9382'
    from fast_delta_rule.runtime import check_baseline
    baseline_commit=check_baseline()
    stop=threading.Event();monitor_errors=[]
    watcher=threading.Thread(target=monitor,args=(stop,monitor_errors),daemon=True);watcher.start()
    frozen=snapshot()
    (OUT/'manifest.json').write_text(json.dumps(frozen,indent=2))
    with zipfile.ZipFile(OUT/'sources.zip','w',zipfile.ZIP_DEFLATED,strict_timestamps=False) as archive:
        for name in frozen:
            archive.write(name,hashlib.sha256(str(Path(name).parent).encode()).hexdigest()[:12]+'/'+Path(name).name)
    emit('environment',device=torch.npu.get_device_name(0),visible_devices=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),npu_smi_card=SMI_CARD,npu_smi_chip=SMI_CHIP,torch=torch.__version__,baseline_commit=baseline_commit,
         torch_npu=torch_npu.__version__,args=vars(a),health=health(),
         policy='native recompute; both variants identical forward saves',
         timing='synchronized eager wall; isolated backward measured separately')
    ad.install(a.family,a.gdn2_variant,a.kda_variant)
    m,info=build(a)
    emit('model',**info)
    initial={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
    m=m.npu()
    torch.manual_seed(20261028)
    x=torch.randint(0,32000,(a.batch,a.tokens),device='npu')
    # Identical full-network initial loss/gradient comparison, not a convergence test.
    refs={};metrics=[];initial_losses={}
    for variant in ('original','optimized'):
        ad.VARIANT=variant
        m.zero_grad(set_to_none=True)
        z=loss(m,x);z.backward();torch.npu.synchronize()
        initial_losses[variant]=float(z)
        assert torch.isfinite(z).item() and finite_model(m)
        assert ad.COUNTS[variant+'_backward']==info['layers'],ad.COUNTS
        if variant=='original':
            refs={n:p.grad.detach().cpu().clone() for n,p in m.named_parameters() if p.grad is not None}
        else:
            for n,p in m.named_parameters():
                if p.grad is None:
                    assert n not in refs
                    continue
                g=p.grad.detach().cpu().float();r=refs.pop(n).float();d=g-r
                rel=float(d.norm()/r.norm().clamp_min(1e-12));mx=float(d.abs().max())
                metrics.append({'name':n,'rel_l2':rel,'max_abs':mx,'passes_008':rel<=.008 or mx<=1e-6})
            assert not refs
    assert initial_losses['original']==initial_losses['optimized'],initial_losses
    (OUT/'gradient-diagnostics.json').write_text(json.dumps(metrics,indent=2))
    emit('validation',loss=initial_losses,all_finite=True,forward_loss_equal=True,
         gradient_max_rel=max(z['rel_l2'] for z in metrics),
         gradient_items_over_008=sum(not z['passes_008'] for z in metrics),
         counts=dict(ad.COUNTS),trace=ad.TRACE,
         numerical_status='diagnostic only; no full precision or convergence acceptance claimed')
    del refs,metrics,z
    m.zero_grad(set_to_none=True);gc.collect();torch.npu.empty_cache()
    if a.smoke:
        a.samples=1;a.warmup=1
    samples=collections.defaultdict(list)
    for block,variant in enumerate(('original','optimized','optimized','original')):
        ad.VARIANT=variant
        m.zero_grad(set_to_none=True);m.load_state_dict(initial)
        optimizer=torch.optim.AdamW(m.parameters(),lr=1e-4,betas=(.9,.95),weight_decay=.1,foreach=True)
        def step():
            optimizer.zero_grad(set_to_none=True)
            z=loss(m,x);z.backward();optimizer.step()
            return z
        for _ in range(a.warmup):z=step()
        torch.npu.synchronize()
        assert torch.isfinite(z).item() and finite_model(m)
        emit('block_ready',block=block,variant=variant,loss=float(z))
        for i in range(a.samples):
            before=ad.COUNTS[variant+'_backward']
            torch.npu.synchronize();start=time.perf_counter()
            z=step();torch.npu.synchronize();ms=(time.perf_counter()-start)*1000
            assert ad.COUNTS[variant+'_backward']-before==info['layers']
            samples[variant].append(ms)
            emit('sample',boundary='STEP',variant=variant,block=block,index=i,ms=ms)
        assert torch.isfinite(z).item() and finite_model(m)
        for st in optimizer.state.values():
            assert torch.isfinite(st['exp_avg']).all().item() and torch.isfinite(st['exp_avg_sq']).all().item()
            assert int(st['step'])==a.warmup+a.samples
        last_name,last_param=list(m.named_parameters())[-1]
        update=float((last_param.detach().cpu()-initial[last_name]).norm())
        assert update>0,'optimizer performed no update'
        emit('optimizer_validation',block=block,variant=variant,steps=a.warmup+a.samples,
             parameter_checked=last_name,parameter_update_norm=update,all_states_finite=True)
        # Isolated full backward and FB: reset same weights and exclude optimizer.
        m.load_state_dict(initial)
        for i in range(max(2,a.samples//2)):
            m.zero_grad(set_to_none=True)
            torch.npu.synchronize();start=time.perf_counter()
            z=loss(m,x);torch.npu.synchronize();mid=time.perf_counter()
            z.backward();torch.npu.synchronize();end=time.perf_counter()
            for boundary,ms in (('B',(end-mid)*1000),('FB',(end-start)*1000)):
                samples[boundary+':'+variant].append(ms)
                emit('sample',boundary=boundary,variant=variant,block=block,index=i,ms=ms)
        emit('block_complete',block=block,variant=variant,finite=finite_model(m),
             allocated_peak=torch.npu.max_memory_allocated(),counts=dict(ad.COUNTS))
        del optimizer,z
        m.zero_grad(set_to_none=True);gc.collect();torch.npu.empty_cache()
    med={k:statistics.median(v) for k,v in samples.items()}
    speed={b:med[p+'original']/med[p+'optimized'] for b,p in [('STEP',''),('B','B:'),('FB','FB:')]}
    assert snapshot()==frozen,'source mutation during experiment'
    stop.set();watcher.join(15)
    assert not watcher.is_alive() and not monitor_errors,monitor_errors
    dependencies={}
    for module in tuple(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename and Path(filename).is_file() and Path(filename).suffix in ('.py','.so'):
            dependencies[filename]=hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    (OUT/'dependencies.json').write_text(json.dumps(dependencies,indent=2))
    emit('summary',speedup=speed,median_ms=med,samples_ms=dict(samples),counts=dict(ad.COUNTS),
         source_hash_verified=True,health=health())
    (OUT/'status.json').write_text(json.dumps({'status':'complete','speedup':speed},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['kda','gdn','gdn2','delta'],required=True)
    p.add_argument('--tokens',type=int,default=512);p.add_argument('--batch',type=int,default=1)
    p.add_argument('--samples',type=int,default=12);p.add_argument('--warmup',type=int,default=5)
    p.add_argument('--tag',required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--gdn2-variant',choices=['residual','combined'],default='residual')
    p.add_argument('--kda-variant',choices=['residual','combined'],default='residual')
    a=p.parse_args();a.heads=a.head_dim=a.value_dim=None;a.workload='standard';a.gdn_kernel='npu_wide'
    OUT=Path(os.environ.get('FDR_MODEL_OUTPUT',str(R/a.tag))).resolve();OUT.mkdir(parents=True,exist_ok=False)
    try:run(a)
    except BaseException as e:
        (OUT/'status.json').write_text(json.dumps({'status':'failed','error':repr(e),'traceback':traceback.format_exc()},indent=2))
        raise
