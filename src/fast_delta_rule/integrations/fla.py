"""Register existing dispatch entries; patch only non-dispatched public aliases."""
import importlib
import sys

_installed=False

def install():
    global _installed
    if _installed: return
    from fla.backends import BaseBackend, register
    import fla.backends as backends
    from fast_delta_rule.runtime import OPERATORS, operator, original_operator
    if backends._DISPATCH_DISABLED:
        raise RuntimeError('Unset FLA_DISABLE_BACKEND_DISPATCH before starting Python.')
    # Capture all originals before installing any aliases or registrations.
    for op in OPERATORS: original_operator(op)
    class FastDeltaBackend(BaseBackend):
        backend_type='fast_delta_rule'
        package_name=None
        env_var=None
        priority=-100
        @classmethod
        def is_available(cls): return True
        def chunk_gated_delta_rule(self,*args,**kwargs): return operator('gdn')(*args,**kwargs)
    kda=importlib.import_module('fla.ops.kda.chunk').chunk_kda
    register(kda,backend=FastDeltaBackend)(operator('kda'))
    register('gated_delta_rule')(FastDeltaBackend)
    for op in ('delta','gdn2'):
        family,name=OPERATORS[op]
        entry=operator(op)
        for path in (f'fla.ops.{family}',f'fla.ops.{family}.chunk'):
            setattr(importlib.import_module(path),name,entry)
        layer=sys.modules.get('fla.layers.'+('delta_net' if op=='delta' else family))
        if layer is not None and hasattr(layer,name): setattr(layer,name,entry)
    _installed=True
