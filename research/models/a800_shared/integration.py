"""Public autograd adapters; all non-target model operations stay unchanged."""
import contextlib
import importlib
import inspect
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scalar'))
import common
import torch
from kernels.kda_residual_local_beta import LocalBetaResidual
from kernels.kda_intra_beta_masked import chunk_kda_bwd_intra as residual_intra
from endpoint_intra import EndpointMatrixIntra
from scalar_adjoint import backward as scalar_backward
from gdn2_consumer_fp32 import residual as gdn2_producer, matrix_intra as gdn2_consumer
from vector_state_gpu import state_bwd as gdn2_state

K=importlib.import_module('fla.ops.kda.chunk_bwd')
G=importlib.import_module('fla.ops.gated_delta_rule.chunk')
D=importlib.import_module('fla.ops.gdn2.chunk_bwd')
BASE_GDN=G.chunk_gated_delta_rule_bwd
G_SIG=inspect.signature(BASE_GDN)
MATRIX=EndpointMatrixIntra(K.chunk_kda_bwd_intra,residual_intra,True)
WY=LocalBetaResidual()
COUNTS={}
AUDIT_LOCAL=False
AUDITS=[]
BASES=[(K,'chunk_kda_bwd_kernel_wy_dqkg_fused',K.chunk_kda_bwd_kernel_wy_dqkg_fused),
       (K,'chunk_kda_bwd_intra',K.chunk_kda_bwd_intra),
       (G,'chunk_gated_delta_rule_bwd',BASE_GDN),
       (D,'chunk_gdn2_bwd_wy_dqkg_fused',D.chunk_gdn2_bwd_wy_dqkg_fused),
       (D,'chunk_gdn2_bwd_intra',D.chunk_gdn2_bwd_intra),
       (D,'chunk_gated_delta_rule_bwd_dhu',D.chunk_gated_delta_rule_bwd_dhu)]
PUBLIC={op:importlib.import_module('fla.ops.'+name+'.chunk') for op,name in [('kda','kda'),('gdn2','gdn2')]}
RAW={op:getattr(mod,'chunk_'+op+'_bwd') for op,mod in PUBLIC.items()}

def audit_results(op,ref,got):
    names={'kda':['dq','dk','dv','dbeta','dg','dh0','dA_log','ddt_bias'],
           'gdn':['dq','dk','dv','dbeta','dg','dh0','dA_log','ddt_bias'],
           'gdn2':['dq','dk','dv','db','dw','dg','dh0','dA_log','ddt_bias']}[op]
    ms={}
    for n,a,b in zip(names,ref,got):
        if a is None or b is None:
            assert a is None and b is None,(op,n);continue
        x=a.double();y=b.double();err=(y-x);norm=x.norm().item();maximum=x.abs().max().item()
        rel=err.norm().item()/max(norm,1e-30);absolute=err.abs().max().item()
        limit=.02 if n in ('dbeta','db','dw','dg','dA_log','ddt_bias') else .008
        passed=bool(torch.isfinite(y).all()) and (absolute<=1e-6 if maximum<1e-6 else rel<=limit)
        ms[n]=dict(relative_l2=rel,max_abs=absolute,limit=limit,passed=passed)
    row=dict(op=op,index=len(AUDITS),metrics=ms,passed=all(m['passed'] for m in ms.values()))
    AUDITS.append(row)
    assert row['passed'],row

def audited_backward(op,*args,**kwargs):
    with implementation(op,'original'):ref=RAW[op](*args,**kwargs)
    got=RAW[op](*args,**kwargs)
    audit_results(op,ref,got)
    return got

def count(name):COUNTS[name]=COUNTS.get(name,0)+1

def gdn_backward(*args,**kwargs):
    count('gdn_optimized_backward')
    a=G_SIG.bind(*args,**kwargs);a.apply_defaults();a=a.arguments
    assert a['chunk_size']==64 and a['cp_context'] is None
    assert not a['use_gate_in_kernel'] and a['cu_seqlens'] is None
    q,k,v,g,b=a['q'],a['k'],a['v'],a['g'],a['beta']
    assert a['scale']==k.shape[-1]**-.5
    op=SimpleNamespace(c=G,gated=True,sequence={})
    xs=(q,k,v,g,b,a['initial_state'])
    # Native backward receives the cumulative gate and triangular A.
    fw=((g,None,a['A']),{},None)
    dq,dk,dv,dg,db,dh=scalar_backward(op,xs,(a['do'],a['dht']),fw,a['state_v_first'],variant='liveness')
    got=(dq,dk,dv,db,dg,dh,None,None)
    if AUDIT_LOCAL:audit_results('gdn',BASE_GDN(*args,**kwargs),got)
    return got

@contextlib.contextmanager
def implementation(op,variant):
    assert variant in ('original','optimized')
    saved=[]
    def patch(module,name,fn):saved.append((module,name,getattr(module,name)));setattr(module,name,fn)
    if variant=='original':
        for module,name,fn in BASES:patch(module,name,fn)
        for key,module in PUBLIC.items():patch(module,'chunk_'+key+'_bwd',RAW[key])
    if variant=='optimized':
        if op=='kda':
            patch(K,'chunk_kda_bwd_kernel_wy_dqkg_fused',WY)
            def intra(**kw):count('kda_optimized_backward');return MATRIX(**kw)
            patch(K,'chunk_kda_bwd_intra',intra)
        elif op=='gdn':patch(G,'chunk_gated_delta_rule_bwd',gdn_backward)
        else:
            patch(D,'chunk_gdn2_bwd_wy_dqkg_fused',gdn2_producer)
            def intra(**kw):count('gdn2_optimized_backward');return gdn2_consumer(**kw)
            patch(D,'chunk_gdn2_bwd_intra',intra)
            patch(D,'chunk_gated_delta_rule_bwd_dhu',gdn2_state)
        if AUDIT_LOCAL and op in PUBLIC:
            patch(PUBLIC[op],'chunk_'+op+'_bwd',lambda *a,**kw:audited_backward(op,*a,**kw))
    try:yield
    finally:
        for module,name,fn in reversed(saved):setattr(module,name,fn)

def backend_snapshot():
    return dict(counts=dict(COUNTS),kda_wrapper_fallback_calls=MATRIX.fallback_calls,
                kda_last_flags=None if MATRIX.flags is None else dict(fallback=int(MATRIX.flags.sum()),total=MATRIX.flags.numel()))
