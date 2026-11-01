"""Fast delta-rule operators. No framework imports or patches on import."""
from .runtime import use_backend, stats, assert_used

__version__ = '0.1.0'

def enable(backend='optimized', *, strict=True, ops=None, integrations=('fla',)):
    from .integrations.fla import install
    from .runtime import configure, check_baseline
    check_baseline()
    install()
    for name in integrations:
        if name=='fla': continue
        if name=='qwen3_5':
            from .integrations.models import qwen3_5
            qwen3_5()
        elif name=='gdn2':
            from .integrations.models import gdn2
            gdn2()
        else: raise ValueError(f'Unknown integration: {name}')
    configure(backend,strict=strict,ops=ops)

def disable():
    from .runtime import configure
    configure('original')

def chunk_kda(*args,**kwargs):
    from .runtime import operator
    return operator('kda')(*args,**kwargs)

def chunk_gated_delta_rule(*args,**kwargs):
    from .runtime import operator
    return operator('gdn')(*args,**kwargs)

def chunk_gdn2(*args,**kwargs):
    from .runtime import operator
    return operator('gdn2')(*args,**kwargs)

def chunk_delta_rule(*args,**kwargs):
    from .runtime import operator
    return operator('delta')(*args,**kwargs)
