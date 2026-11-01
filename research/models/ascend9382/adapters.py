"""Public FLA interface, common Ascend compatibility, measured backward switch.

Both variants use identical original forward, public gate/norm transforms and
recomputation. The zero initial state is the mathematical no-cache boundary.
K-first state layout avoids the frozen V-first UB limitation on this device.
"""
from collections import Counter
import functools, importlib, inspect
import torch
from fast_delta_rule.backends.backward import _clone
VARIANT = 'original'
COUNTS = Counter()
TRACE = {}

def install(family, gdn2_variant='residual', kda_variant='residual'):
    global VARIANT
    path = {'kda': 'kda', 'gdn': 'gated_delta_rule', 'gdn2': 'gdn2', 'delta': 'delta_rule'}[family]
    entry = {'kda': 'chunk_kda', 'gdn': 'chunk_gated_delta_rule', 'gdn2': 'chunk_gdn2', 'delta': 'chunk_delta_rule'}[family]
    c = importlib.import_module('fla.ops.' + path + '.chunk')
    raw_public = inspect.unwrap(getattr(c, entry))
    b = importlib.import_module('fla.ops.' + path + '.chunk_bwd')
    if family == 'gdn2':
        from gdn2_original_intra import intra
        b.chunk_gdn2_bwd_intra = intra
        from fla.ops.gdn2.backends.triton_ascend.chunk_bwd import chunk_gdn2_bwd_wy_dqkg_fused_npu
        b.chunk_gdn2_bwd_wy_dqkg_fused = chunk_gdn2_bwd_wy_dqkg_fused_npu
        from fast_delta_rule.kernels.gdn2_full_npu import residual
        overrides = {'chunk_gdn2_bwd_wy_dqkg_fused': residual}
        if gdn2_variant == 'combined':
            from matrix_final_npu import gdn2_intra

            def intra_boundary(**kw):
                out = gdn2_intra(**kw)
                return (out[0].to(kw['q']), out[1].to(kw['k']), out[2], out[3])
            overrides['chunk_gdn2_bwd_intra'] = intra_boundary
    else:
        from fast_delta_rule.kernels import kda_residual_npu_onchip as residual
        residual._CORE_BK = 128
        overrides = dict(chunk_kda_bwd_wy_dqkg_fused=residual.chunk_kda_bwd_wy_dqkg_fused_npu)
        if kda_variant == 'combined':
            from matrix_final_npu import kda_intra
            overrides['chunk_kda_bwd_intra'] = kda_intra
    bwdname = entry + '_bwd'
    original = getattr(b, bwdname)
    optimized = _clone(original, overrides)
    sig = inspect.signature(original)

    @functools.wraps(original)
    def backward(*args, **kw):
        bound = sig.bind(*args, **kw)
        bound.apply_defaults()
        a = dict(bound.arguments)
        if a['dht'] is None:
            a['dht'] = torch.zeros_like(a['initial_state'])
        COUNTS[VARIANT + '_backward'] += 1
        out = (original if VARIANT == 'original' else optimized)(**a)
        key = family + '_' + VARIANT + '_backward'
        if key not in TRACE:
            TRACE[key] = {'inputs': {n: {'shape': list(x.shape), 'dtype': str(x.dtype)} for n, x in a.items() if torch.is_tensor(x)}, 'outputs': [{'shape': list(x.shape), 'dtype': str(x.dtype)} if torch.is_tensor(x) else None for x in out]}
            if VARIANT == 'optimized':
                assert TRACE[key]['outputs'] == TRACE[family + '_original_backward']['outputs'], TRACE[key]
        return out
    setattr(c, bwdname, backward)

    @functools.wraps(raw_public)
    def public(*args, **kw):
        bound = inspect.signature(raw_public).bind(*args, **kw)
        bound.apply_defaults()
        a = dict(bound.arguments)
        a.update(a.pop('kwargs', {}))
        q, k, v = (a['q'], a['k'], a['v'])
        assert q.device.type == 'npu' and q.dtype == torch.bfloat16
        assert a.get('cu_seqlens') is None and a.get('initial_state') is None
        assert not a['output_final_state']
        if family not in TRACE:
            TRACE[family] = {n: {'shape': list(x.shape), 'dtype': str(x.dtype)} for n, x in a.items() if torch.is_tensor(x)}
        a['initial_state'] = torch.zeros((k.shape[0], v.shape[2], k.shape[-1], v.shape[-1]), device=k.device, dtype=torch.float32)
        a['output_final_state'] = True
        a['state_v_first'] = False
        COUNTS[VARIANT + '_forward'] += 1
        out, _ = raw_public(**a)
        return (out, None)
    for mod in (c, importlib.import_module('fla.ops.' + path)):
        setattr(mod, entry, public)
    layer = importlib.import_module('fla.layers.' + {'delta': 'delta_net', 'gdn': 'gated_deltanet'}.get(family, family))
    setattr(layer, entry, public)
    VARIANT = 'original'
