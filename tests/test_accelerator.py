"""Run explicitly with FDR_TEST_DEVICE=cuda (or npu on the documented target)."""
import os
import pytest

pytestmark = [pytest.mark.accelerator,pytest.mark.skipif(not os.getenv('FDR_TEST_DEVICE'),reason='accelerator tests are opt-in')]

@pytest.mark.parametrize('op',['kda','gdn','gdn2','delta'])
def test_public_integration_and_graph_binding(op):
    import importlib
    import torch
    import fast_delta_rule as fdr
    from fast_delta_rule.runtime import OPERATORS,original_operator
    from fast_delta_rule.validation import inputs,error
    device=os.environ['FDR_TEST_DEVICE']
    family,name=OPERATORS[op]
    original=original_operator(op)
    # Install before importing layer aliases, as required by the launcher.
    fdr.enable(ops=(op,))
    entry=getattr(importlib.import_module('fla.ops.'+family),name)
    xs=inputs(op,device,length=128)
    do=torch.randn_like(xs['v']);dh=torch.randn_like(xs['initial_state'])
    ref=original(**xs,output_final_state=True)
    rg=torch.autograd.grad(ref,tuple(xs.values()),(do,dh))
    fdr.stats(reset=True)
    out=entry(**xs,output_final_state=True)
    fdr.disable()
    cg=torch.autograd.grad(out,tuple(xs.values()),(do,dh))
    fdr.assert_used((op,))
    for n,r,c in zip(xs,rg,cg):
        assert error(r,c,.02 if n in ('g','beta','b','w') else .008)['passed'],n
    assert not any('fallback' in k or 'rejected' in k for k in fdr.stats())

@pytest.mark.parametrize('op',['kda','gdn','gdn2','delta'])
def test_absent_state_and_noncontiguous_input(op):
    import torch
    import fast_delta_rule as fdr
    from fast_delta_rule.runtime import original_operator,operator
    from fast_delta_rule.validation import inputs,error
    xs=inputs(op,os.environ['FDR_TEST_DEVICE'],length=128)
    xs.pop('initial_state')
    xs['q']=xs['q'].transpose(1,2).contiguous().transpose(1,2).detach().requires_grad_()
    do=torch.randn_like(xs['v'])
    ref,_=original_operator(op)(**xs)
    rg=torch.autograd.grad(ref,tuple(xs.values()),do)
    with fdr.use_backend('optimized',ops=(op,)):out,_=operator(op)(**xs)
    cg=torch.autograd.grad(out,tuple(xs.values()),do)
    for n,r,c in zip(xs,rg,cg):
        assert error(r,c,.02 if n in ('g','beta','b','w') else .008)['passed'],n

def test_rejection_and_visible_fallback():
    import torch
    import fast_delta_rule as fdr
    from fast_delta_rule.validation import inputs
    xs=inputs('kda',os.environ['FDR_TEST_DEVICE'],length=65)
    with fdr.use_backend('optimized'):
        with pytest.raises(RuntimeError,match='multiple of 64'):fdr.chunk_kda(**xs)
    fdr.stats(reset=True)
    with fdr.use_backend('optimized',strict=False):
        with pytest.warns(RuntimeWarning,match='using original'):
            out,_=fdr.chunk_kda(**xs)
    out.sum().backward()
    assert fdr.stats()['kda.fallback_forward']==1
    assert fdr.stats().get('kda.optimized_backward',0)==0

def test_kda_model_gate_gradients():
    import torch
    import fast_delta_rule as fdr
    from fast_delta_rule.runtime import original_operator
    from fast_delta_rule.validation import inputs,error
    xs=inputs('kda',os.environ['FDR_TEST_DEVICE'],length=128)
    xs['g']=torch.randn_like(xs['q']).requires_grad_()
    xs['beta']=torch.randn_like(xs['beta']).requires_grad_()
    xs['A_log']=torch.full((2,),-2.,device=xs['q'].device,requires_grad=True)
    xs['dt_bias']=torch.zeros((256,),device=xs['q'].device,requires_grad=True)
    kw=dict(xs,use_gate_in_kernel=True,use_beta_sigmoid_in_kernel=True,
            use_qk_l2norm_in_kernel=True,state_v_first=True)
    ref,_=original_operator('kda')(**kw)
    do=torch.randn_like(ref)
    rg=torch.autograd.grad(ref,tuple(xs.values()),do)
    with fdr.use_backend('optimized'):out,_=fdr.chunk_kda(**kw)
    cg=torch.autograd.grad(out,tuple(xs.values()),do)
    for n,r,c in zip(xs,rg,cg):
        assert error(r,c,.02 if n in ('g','beta','A_log','dt_bias') else .008)['passed'],n
