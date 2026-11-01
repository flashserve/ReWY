"""Uniform family backends, preserving FLA public gate/norm autograd."""
import contextlib, importlib, inspect, sys
from pathlib import Path
from types import SimpleNamespace
BASE=Path(__file__).resolve().parent.parent/'a800_shared'
sys.path.insert(0,str(BASE))
import integration as legacy
import torch
import scalar_liveness_gpu
from scalar_adjoint import backward as scalar_backward
from gdn_wide64 import adjoint as gdn_adjoint
AUDITS=legacy.AUDITS
COUNTS=legacy.COUNTS
AUDIT_LOCAL=False
MATRIX=legacy.MATRIX
G=legacy.G
D=importlib.import_module('fla.ops.delta_rule.chunk')
GRAW=G.chunk_gated_delta_rule_bwd
DRAW=D.chunk_delta_rule_bwd
GSIG=inspect.signature(GRAW);DSIG=inspect.signature(DRAW)
LIVENESS=scalar_liveness_gpu.adjoint

def audit(op,ref,got):
    metrics=[]
    for x,y in zip(ref,got):
        if x is None or y is None:
            assert x is None and y is None
            continue
        assert x.shape==y.shape and x.dtype==y.dtype
        assert bool(torch.isfinite(y).all())
        a=x.double();b=y.double();d=b-a
        metrics.append(dict(relative_l2=d.norm().item()/max(a.norm().item(),1e-30),max_abs=d.abs().max().item(),finite=True))
    AUDITS.append(dict(op=op,index=len(AUDITS),metrics=metrics,passed=True,precision_enforced=False))

def gdn_backward(*args,**kwargs):
    legacy.count('gdn_optimized_backward')
    bind=GSIG.bind(*args,**kwargs);bind.apply_defaults();a=bind.arguments
    assert a['chunk_size']==64 and a['cp_context'] is None and a['cu_seqlens'] is None
    q,k,v,g,b=(a[n] for n in ('q','k','v','g','beta'))
    assert a['scale']==k.shape[-1]**-.5
    xs=(q,k,v,g,b,a['initial_state'])
    old=scalar_liveness_gpu.adjoint;scalar_liveness_gpu.adjoint=gdn_adjoint
    try: dq,dk,dv,dg,db,dh=scalar_backward(SimpleNamespace(c=G,gated=True,sequence={}),xs,(a['do'],a['dht']),((g,None,a['A']),{},None),a['state_v_first'],variant='liveness')
    finally: scalar_liveness_gpu.adjoint=old
    da,dt=None,None
    if a['use_gate_in_kernel']:
        dg,da,dt=G.gdn_gate_bwd(g=a['g_input'],A_log=a['A_log'],dt_bias=a['dt_bias'],dyg=dg)
    got=(dq,dk,dv,db,dg,dh,da,dt)
    if AUDIT_LOCAL:audit('gdn',GRAW(*args,**kwargs),got)
    return got

def delta_backward(*args,**kwargs):
    legacy.count('delta_optimized_backward')
    bind=DSIG.bind(*args,**kwargs);bind.apply_defaults();a=bind.arguments
    assert a['chunk_size']==64 and a['cu_seqlens'] is None
    q,k,v,b=(a[n] for n in ('q','k','v','beta'))
    assert a['scale']==k.shape[-1]**-.5
    got=scalar_backward(SimpleNamespace(c=D,gated=False,sequence={}),
        (q,k,v,b,a['initial_state']),(a['do'],a['dht']),((None,a['A']),{},None),variant='liveness')
    if AUDIT_LOCAL:audit('delta',DRAW(*args,**kwargs),got)
    return got

@contextlib.contextmanager
def implementation(family,variant):
    legacy.AUDIT_LOCAL=AUDIT_LOCAL
    old_audit=legacy.audit_results;legacy.audit_results=audit
    if family=='gdn2':
        try:
            with legacy.implementation(family,variant):yield
        finally:legacy.audit_results=old_audit;legacy.AUDIT_LOCAL=False
    else:
        module,name,raw,optimized=(G,'chunk_gated_delta_rule_bwd',GRAW,gdn_backward) if family=='gdn' else (D,'chunk_delta_rule_bwd',DRAW,delta_backward)
        saved=getattr(module,name);setattr(module,name,raw if variant=='original' else optimized)
        try:yield
        finally:setattr(module,name,saved);legacy.audit_results=old_audit;legacy.AUDIT_LOCAL=False

def backend_snapshot():return dict(counts=dict(COUNTS))
