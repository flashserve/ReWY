"""Construct complete published/default architectures without pretrained weights."""
import ast
import functools
import importlib
import importlib.util
import json
from pathlib import Path
import sys
import types
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
ROOT = Path(__file__).resolve().parent
TRACE = {}

def traced(name, fn):

    def call(*args, **kw):
        if name not in TRACE:
            values = dict(zip(('q', 'k', 'v'), args))
            values.update(kw)
            TRACE[name] = {k: dict(shape=list(v.shape), dtype=str(v.dtype), stride=list(v.stride())) for k, v in values.items() if torch.is_tensor(v)}
            TRACE[name]['options'] = {k: v for k, v in kw.items() if isinstance(v, (bool, int, float)) or v is None}
        return fn(*args, **kw)
    return call

def make_model(op, seed=20261027):
    torch.manual_seed(seed)
    from fla.models.kda import KDAConfig, KDAForCausalLM
    import fla.layers.kda as layer
    if not getattr(layer.chunk_kda, '_e2e_traced', False):
        layer.chunk_kda = traced('kda', layer.chunk_kda)
        layer.chunk_kda._e2e_traced = True
    cfg = KDAConfig(use_cache=False)
    model = KDAForCausalLM(cfg)
    info = dict(model='FLA KDAConfig default causal LM', config=cfg.to_dict(), layers=24, target_layers=24, vocab_size=32000)
    model.train()
    assert all((p.dtype == torch.float32 for p in model.parameters()))
    info['parameter_dtype'] = 'torch.float32'
    info['parameters'] = sum((p.numel() for p in model.parameters()))
    info['trainable_parameters'] = sum((p.numel() for p in model.parameters() if p.requires_grad))
    return (model, info)

def loss_for(model, op, tokens, labels):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return model(input_ids=tokens, labels=labels, use_cache=False).loss
