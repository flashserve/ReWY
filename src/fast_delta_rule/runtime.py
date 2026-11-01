"""Explicit selection and auditing. Importing this module never patches FLA."""
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache, wraps
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import warnings

OPERATORS = {
    'kda': ('kda', 'chunk_kda'),
    'gdn': ('gated_delta_rule', 'chunk_gated_delta_rule'),
    'gdn2': ('gdn2', 'chunk_gdn2'),
    'delta': ('delta_rule', 'chunk_delta_rule'),
}

@dataclass(frozen=True)
class Selection:
    backend: str = 'original'
    strict: bool = True
    ops: tuple = tuple(OPERATORS)

_selection = ContextVar('fast_delta_rule_selection', default=Selection())
_counts = Counter()
_warned = set()

def record(op, event):
    _counts[f'{op}.{event}'] += 1

def stats(reset=False):
    result = dict(sorted(_counts.items()))
    if reset: _counts.clear()
    return result

def configure(backend='optimized', *, strict=True, ops=None):
    if backend not in ('original', 'optimized'): raise ValueError('backend must be original or optimized')
    ops = tuple(OPERATORS) if ops is None else tuple(ops)
    unknown = set(ops)-OPERATORS.keys()
    if unknown: raise ValueError(f'Unknown operators: {sorted(unknown)}')
    return _selection.set(Selection(backend, strict, ops))

@contextmanager
def use_backend(backend, *, strict=True, ops=None):
    token = configure(backend, strict=strict, ops=ops)
    try: yield
    finally: _selection.reset(token)

@lru_cache(None)
def check_baseline():
    spec = importlib.util.find_spec('fla')
    if spec is None or spec.origin is None:
        raise RuntimeError('FLA is missing. Install the frozen dependency with scripts/install.sh cuda|npu.')
    root = Path(spec.origin).parent.parent
    expected = json.loads(Path(__file__).with_name('baseline.json').read_text())
    mismatches = []
    for name, digest in expected['files'].items():
        path = root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=digest:
            mismatches.append(name)
    if mismatches:
        raise RuntimeError(f"Unsupported FLA sources; require {expected['commit']}. "
                           f"First mismatches: {mismatches[:4]}. Install in a dedicated environment.")
    return expected['commit']

@lru_cache(None)
def original_operator(op):
    family, name = OPERATORS[op]
    # Unwrap dispatch so 'original' cannot accidentally select our registration.
    return inspect.unwrap(getattr(importlib.import_module(f'fla.ops.{family}.chunk'), name))

def rejection_reason(op, arguments):
    """The release envelope is capability based, never benchmark-case based."""
    import torch
    q,k,v = (arguments.get(n) for n in ('q','k','v'))
    if not all(isinstance(x,torch.Tensor) and x.ndim==4 for x in (q,k,v)):
        return 'expected rank-4 q/k/v'
    if q.device.type not in ('cuda','npu'): return 'CUDA or Ascend required'
    if any(x.device!=q.device for x in (k,v)): return 'mixed devices'
    if any(x.dtype!=torch.bfloat16 for x in (q,k,v)): return 'release envelope uses BF16 q/k/v'
    if k.shape[-1]!=128 or v.shape[-1]!=128: return 'release envelope requires K=V=128'
    if q.shape!=k.shape or k.shape[:3]!=v.shape[:3]: return 'grouped heads or unequal q/k/v shapes'
    if k.shape[1]<64 or k.shape[1]%64: return 'sequence length must be a positive multiple of 64'
    if arguments.get('chunk_size',64)!=64: return 'release envelope requires chunk_size=64'
    if arguments.get('cu_seqlens') is not None: return 'packed/variable sequences are not release-validated'
    if arguments.get('cp_context') is not None: return 'context parallelism is not release-validated'
    if arguments.get('safe_gate',False): return 'safe_gate transform is not release-validated'
    if arguments.get('use_graph',False): return 'static variable-length graph buffers are not supported'
    if arguments.get('return_intermediate_states',False): return 'intermediate inference states are not supported'
    scale=arguments.get('scale')
    if scale is not None and scale!=k.shape[-1]**-.5: return 'custom scale is not release-validated'
    if op=='gdn' and arguments.get('use_gate_in_kernel',False): return 'GDN fused gate is not release-validated'
    if op=='gdn2' and arguments.get('use_gate_in_kernel',False): return 'GDN2 fused gate is not release-validated'
    if op=='delta' and arguments.get('head_first',False): return 'head-first layout is not supported'
    if q.device.type=='npu':
        import os
        if os.environ.get('FDR_EXPERIMENTAL_NPU')!='1':
            return 'NPU packaging awaits hardware revalidation; set FDR_EXPERIMENTAL_NPU=1 only for validation'
        if arguments.get('initial_state') is None: return 'Ascend release requires explicit initial_state'
        if not arguments.get('output_final_state',False): return 'Ascend release requires output_final_state=True'
        from fast_delta_rule.backends.npu_target import device_backend
        device_backend(q.device)  # Reject unreviewed physical devices.
    return None

@lru_cache(None)
def optimized_operator(op):
    check_baseline()
    module = importlib.import_module(f'fast_delta_rule._compat.{op}')
    return getattr(module,OPERATORS[op][1])

@lru_cache(None)
def operator(op):
    if op not in OPERATORS: raise ValueError(f'Unknown operator {op!r}')
    base=original_operator(op)
    signature=inspect.signature(base)
    @wraps(base)
    def call(*args,**kwargs):
        config=_selection.get()
        if config.backend=='original' or op not in config.ops:
            record(op,'original_forward')
            return base(*args,**kwargs)
        bound=signature.bind(*args,**kwargs)
        bound.apply_defaults()
        arguments=dict(bound.arguments)
        # FLA carries A_log/dt_bias and a few legacy options through **kwargs.
        arguments.update(arguments.pop('kwargs',{}))
        reason=rejection_reason(op,arguments)
        if reason:
            record(op,'rejected:'+reason)
            if config.strict: raise RuntimeError(f'fast-delta-rule {op}: {reason}')
            key=(op,reason)
            if key not in _warned:
                warnings.warn(f'fast-delta-rule {op} using original: {reason}',RuntimeWarning,stacklevel=2)
                _warned.add(key)
            record(op,'fallback_forward')
            return base(*args,**kwargs)
        result=optimized_operator(op)(*args,**kwargs)
        record(op,'optimized_forward')
        return result
    return call

def assert_used(ops=None, *, backward=True):
    ops=_selection.get().ops if ops is None else ops
    suffix='optimized_backward' if backward else 'optimized_forward'
    missing=[op for op in ops if not _counts.get(f'{op}.{suffix}',0)]
    if missing: raise RuntimeError(f'No {suffix} calls for {missing}; check integration and whether backward ran.')
