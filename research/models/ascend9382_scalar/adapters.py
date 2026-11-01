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
    from fast_delta_rule.kernels.gdn_original_separate_npu import prepare_wy_repr_bwd_npu, recompute_w_u_fwd_npu
    c.recompute_w_u_fwd = recompute_w_u_fwd_npu
    if family == 'gdn':
        c.prepare_wy_repr_bwd = prepare_wy_repr_bwd_npu
    else:
        from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
        c.prepare_wy_repr_fwd = chunk_gated_delta_rule_fwd_intra
        c.prepare_wy_repr_bwd = lambda **kw: prepare_wy_repr_bwd_npu(**kw)[:3]
    bwdname = entry + '_bwd'
    original = getattr(c, bwdname)
    from fast_delta_rule.backends.backward import _scalar

    def optimized(**kw):
        out = _scalar(family, (), kw)
        if family == 'gdn' and kw.get('use_gate_in_kernel'):
            dq, dk, dv, db, dg, dh, _, _ = out
            dg, da, dt = c.gdn_gate_bwd(g=kw['g_input'], A_log=kw['A_log'], dt_bias=kw['dt_bias'], dyg=dg)
            out = (dq, dk, dv, db, dg, dh, da, dt)
        return out
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
        if family != 'delta':
            a['state_v_first'] = False
        COUNTS[VARIANT + '_forward'] += 1
        out, _ = raw_public(**a)
        return (out, None)
    for mod in (c, importlib.import_module('fla.ops.' + path)):
        setattr(mod, entry, public)
    layer = importlib.import_module('fla.layers.' + {'delta': 'delta_net', 'gdn': 'gated_deltanet'}.get(family, family))
    setattr(layer, entry, public)
    VARIANT = 'original'
