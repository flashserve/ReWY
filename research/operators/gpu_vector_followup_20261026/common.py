"""Shared experiment infrastructure, isolated from earlier mutable harnesses."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import zipfile
import time
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.append(str(ROOT/'family_completion_20261024'))
import run_cuda as e
import torch
import triton

DEV, ACC = e.DEVICE, e.ACCEL
EMIT = e.emit
spec = importlib.util.spec_from_file_location('prior_timing', ROOT/'delta_algebra_20261024/common.py')
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)
capture, timing = prior.capture, prior.timing
CASES = prior.CASES

def eager_timing(functions,rounds=10):
    """Host-synchronized FB boundary for upstream functions with host barriers.

    This is explicitly not a graph/device-only measurement. Keep the same
    paired order, two-call unroll and 20 observations per implementation.
    """
    names=list(functions);start=time.perf_counter()
    while time.perf_counter()-start<2:
        for n in names:functions[n]()
        ACC.synchronize()
    start=time.perf_counter()
    for n in names:functions[n]();functions[n]()
    ACC.synchronize();one=(time.perf_counter()-start)/len(names)
    repeat=max(1,min(64,int(.03/max(one,1e-6))))
    samples={n:[] for n in names};orders=[]
    for r in range(rounds):
        shift=(r//2)%len(names);order=names[shift:]+names[:shift]
        if r%2:order.reverse()
        order=order+order[::-1];orders.append(order)
        for n in order:
            ACC.synchronize();start=time.perf_counter()
            for _ in range(repeat):functions[n]();functions[n]()
            ACC.synchronize();samples[n].append((time.perf_counter()-start)*1e6/(repeat*2))
    return dict(samples_us=samples,median_us={n:statistics.median(v) for n,v in samples.items()},orders=orders,iterations=repeat,unroll=2,timing_method='synchronized_eager_wall_clock')

def metrics(ref, actual, names):
    assert len(ref) == len(actual) == len(names)
    return {n:e.error(r,a,.02 if n in ('dg','dbeta','db','dw') else .008)
            for n,r,a in zip(names,ref,actual)}

def check(m):
    return all(v['pass'] for v in m.values())

def hashes():
    files = list(HERE.glob('*.py'))
    manifest = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    tag = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
    folder=HERE/'results';folder.mkdir(exist_ok=True)
    dest=folder/f'sources_{tag}.zip'
    if not dest.exists():
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
            for p in files:z.write(p,p.name)
    return manifest

def emit(x,path):
    EMIT(x,path)

def dependencies():
    files=set()
    for module in tuple(sys.modules.values()):
        value=getattr(module,'__file__',None)
        if not value:continue
        p=Path(value).resolve()
        if p.suffix=='.py' and p.is_relative_to(ROOT) and not p.is_relative_to(HERE) and p.is_file():files.add(p)
    lib=ROOT/'cannbot_npu_20261024/native/build/libkda_residual_native.so'
    if DEV=='npu' and lib.exists():files.add(lib)
    manifest={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    digest=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:16]
    dest=HERE/'results'/f'dependencies_{digest}.zip'
    if not dest.exists():
        with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
            for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
    return dict(sha256=manifest,archive=dest.name)

def launch(kernel, tasks, **kw):
    if DEV == 'npu':
        from fla.utils import ascend_compile_kwargs
        from fla.utils.ascend_ub_manager import get_npu_properties
        nc = get_npu_properties()['num_aicore']
        kernel[(nc,)](TASKS=tasks, STEP=nc, **kw, **ascend_compile_kwargs())
    else:
        kernel[(tasks,)](TASKS=tasks, STEP=tasks, **kw, num_warps=4, num_stages=2)
