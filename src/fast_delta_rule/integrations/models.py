"""Explicit model bindings. No unrelated layers or imports are rewritten."""
import importlib
from fast_delta_rule.runtime import operator

def qwen3_5():
    module=importlib.import_module('transformers.models.qwen3_5.modeling_qwen3_5')
    name='torch_chunk_gated_delta_rule'
    if not hasattr(module,name): raise RuntimeError('Unsupported Qwen3.5 model entry point')
    setattr(module,name,operator('gdn'))

def gdn2():
    # Requires the upstream project and its own dependencies on PYTHONPATH.
    module=importlib.import_module('lit_gpt.gdn2')
    if not hasattr(module,'chunk_gdn2'): raise RuntimeError('Unsupported official GDN2 layer')
    module.chunk_gdn2=operator('gdn2')
