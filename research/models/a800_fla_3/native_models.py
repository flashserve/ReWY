"""Frozen FLA full models; GDN2 ports official dimensions onto FLA's LM shell."""
import importlib
from pathlib import Path
import torch

TRACE={}

def make_model(args):
    from fla.models.gated_deltanet import GatedDeltaNetConfig, GatedDeltaNetForCausalLM
    from fla.models.delta_net import DeltaNetConfig, DeltaNetForCausalLM
    torch.manual_seed(20261027)
    f=args.family
    # Published GDN pure 1.3B and DeltaNet 1.3B; GDN2 official dimensions.
    defaults={'gdn':(2048,24,8,256,None,1e-6),
              'delta':(2048,24,16,128,None,1e-6),
              'gdn2':(2304,18,18,128,6208,1e-5)}
    hidden,layers,h,k,inter,eps=defaults[f]
    heads=args.heads or h;kd=args.head_dim or k;vd=args.value_dim or kd
    cfg_args=dict(hidden_size=hidden,num_hidden_layers=layers,num_heads=heads,
        intermediate_size=inter,norm_eps=eps,vocab_size=32000,use_cache=False,
        attn=None,fuse_cross_entropy=args.workload=='standard',
        fuse_linear_cross_entropy=args.workload=='linear_ce',fuse_swiglu=True,
        tie_word_embeddings=False)
    if f=='delta':
        cfg=DeltaNetConfig(**cfg_args,expand_k=heads*kd/hidden,expand_v=heads*vd/hidden,pad_token_id=2)
        model=DeltaNetForCausalLM(cfg)
        source='fla-hub/delta_net-1.3B-100B/config.json'
        model_name='FLA native DeltaNetForCausalLM'
    else:
        cfg=GatedDeltaNetConfig(**cfg_args,head_dim=kd,expand_v=vd/kd)
        if f=='gdn2':
            from fla.layers.gdn2 import GatedDeltaNet2
            mod=importlib.import_module('fla.models.gated_deltanet.modeling_gated_deltanet')
            raw=mod.GatedDeltaNet
            def factory(**kw):
                for name in ('use_gate','output_gate_activation'):kw.pop(name,None)
                return GatedDeltaNet2(**kw)
            mod.GatedDeltaNet=factory
            try:model=GatedDeltaNetForCausalLM(cfg)
            finally:mod.GatedDeltaNet=raw
            source='NVlabs GatedDeltaNet2 gdn2_1.3B dimensions; FLA GatedDeltaNet2 + common FLA LM shell (port)'
            model_name='FLA GatedDeltaNet2 layer in FLA common LM shell'
        else:
            model=GatedDeltaNetForCausalLM(cfg)
            source='m-a-p/1.3B-100B-GatedDeltaNet-pure/config.json'
            model_name='FLA native GatedDeltaNetForCausalLM'
    model.train()
    assert all(p.dtype==torch.float32 for p in model.parameters())
    base=(heads,kd,vd)==(h,k,k)
    return model,dict(family=f,model=model_name,source=source,config=cfg.to_dict(),
        scope=('published_config' if f!='gdn2' else 'official_dimensions_FLA_port') if base else 'structural_exploration',
        layers=layers,vocab_size=32000,heads=heads,head_dim=kd,value_dim=vd,
        hidden_size=hidden,parameters=sum(p.numel() for p in model.parameters()),parameter_dtype='torch.float32')

def loss_for(model,family,tokens,labels):
    with torch.autocast('cuda',dtype=torch.bfloat16):
        return model(input_ids=tokens,labels=labels,use_cache=False).loss
