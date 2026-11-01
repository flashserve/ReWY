"""Native DeltaNet/GDN orchestration with reversible WY substitutions."""
import importlib
from common import *
from residual_scalar import wy
sys.path.append(str(ROOT / 'delta_algebra_20261024'))

class ScalarOp:

    def __init__(self, name):
        self.name = name
        self.gated = name == 'gdn'
        self.c = importlib.import_module('fla.ops.' + ('gated_delta_rule' if self.gated else 'delta_rule') + '.chunk')
        self.base = self.c.prepare_wy_repr_bwd
        if self.gated:
            module = importlib.import_module('gdn_gram_' + ('gpu' if DEV == 'cuda' else 'npu'))
            self.gram = module.prepare_wy_repr_bwd if DEV == 'cuda' else module.prepare_wy_repr_bwd_npu
        self.names = ['dq', 'dk', 'dv', 'dg', 'dbeta', 'dh0'] if self.gated else ['dq', 'dk', 'dv', 'dbeta', 'dh0']
        self.variants = ['original', 'gram', 'residual_gram', 'residual'] if self.gated else ['original', 'residual']

    def inputs(self, case):
        kw = {k: v for k, v in case.items() if k not in ('zero_beta', 'tiny_beta', 'positive_gate')}
        xs, dys = e.inputs(**kw)
        q, k, v, g, b, h = xs
        if self.gated:
            g = g[..., 0].detach().contiguous().requires_grad_()
        with torch.no_grad():
            if case.get('zero_beta'):
                b[:, ::3] = 0
            if case.get('tiny_beta'):
                b[:, ::3] = 1e-09
            if case.get('positive_gate'):
                g[:, ::3].mul_(-1)
        return ((q, k, v, g, b, h) if self.gated else (q, k, v, b, h), dys)

    def oracle(self, xs, svf=False):
        if self.gated:
            q, k, v, g, b, h = xs
            g = g[..., None].expand(*g.shape, k.shape[-1])
        else:
            q, k, v, b, h = xs
            g = torch.zeros((*v.shape[:3], k.shape[-1]), dtype=k.dtype, device=k.device)
        return e.oracle((q, k, v, g, b, h), svf)

    def forward(self, xs, policy, svf=False):
        c = self.c
        cache = {}
        iname = 'chunk_gated_delta_rule_fwd_intra' if self.gated else 'prepare_wy_repr_fwd'
        actual_xs = xs
        intra, hfn = (getattr(c, iname), c.chunk_gated_delta_rule_fwd_h)

        def save_intra(**kw):
            r = intra(**kw)
            cache['wu'] = r[:2]
            return r

        def save_h(**kw):
            r = hfn(**kw)
            cache['hr'] = (r[0], r[1], None)
            return r
        if policy == 'all':
            setattr(c, iname, save_intra)
            c.chunk_gated_delta_rule_fwd_h = save_h
        try:
            if self.gated:
                q, k, v, g, b, h = actual_xs
                z = c.chunk_gated_delta_rule_fwd(q, k, v, g, b, k.shape[-1] ** (-0.5), h, True, state_v_first=svf, **getattr(self, 'sequence', {}))
                out = (z[1], z[3])
            else:
                q, k, v, b, h = actual_xs
                z = c.chunk_delta_rule_fwd(q, k, v, b, k.shape[-1] ** (-0.5), h, True, **getattr(self, 'sequence', {}))
                out = (z[0], z[2])
            return (z, cache, out)
        finally:
            setattr(c, iname, intra)
            c.chunk_gated_delta_rule_fwd_h = hfn

    def backward(self, xs, dys, fw, variant, svf=False):
        c = self.c
        z, cache, _ = fw
        state = {}
        absent_h = xs[-1] is None
        if 'zero_h0' in cache:
            xs = (*xs[:-1], cache['zero_h0'])
        if variant in ('liveness', 'native_v16s3'):
            from scalar_adjoint import backward
            out = backward(self, xs, dys, fw, svf, variant=variant)
            return (*out[:-1], None) if absent_h else out
        recompute, hfn, oldwy = (c.recompute_w_u_fwd, c.chunk_gated_delta_rule_fwd_h, c.prepare_wy_repr_bwd)
        old_direct = c.chunk_bwd_dqkwg

        def get_h(**kw):
            r = cache['hr'] if 'hr' in cache else hfn(**kw)
            state['h'], state['r'] = r[:2]
            return r

        def residual(**kw):
            impl = wy
            return impl(h=state['h'], r=state['r'], gram=variant in (), state_v_first=svf, **kw)
        c.prepare_wy_repr_bwd = self.base if variant == 'original' else self.gram if variant == 'gram' else residual
        if variant not in ('original',):
            c.chunk_bwd_dqkwg = lambda **kw: old_direct(**dict(kw, w=None))
        if getattr(self, 'record_stage', False):
            stage = c.prepare_wy_repr_bwd

            def recorded(**kw):
                self.last_stage = lambda: stage(**kw)
                return stage(**kw)
            c.prepare_wy_repr_bwd = recorded
        c.chunk_gated_delta_rule_fwd_h = get_h
        if 'wu' in cache:
            c.recompute_w_u_fwd = lambda **kw: cache['wu']
        try:
            if self.gated:
                q, k, v, g, b, h = xs
                dq, dk, dv, db, dg, dh, *_ = c.chunk_gated_delta_rule_bwd(q, k, v, z[0], b, z[2], k.shape[-1] ** (-0.5), h, *dys, state_v_first=svf, **getattr(self, 'sequence', {}))
                return (dq, dk, dv, dg, db, None if absent_h else dh)
            q, k, v, b, h = xs
            out = c.chunk_delta_rule_bwd(q, k, v, b, z[1], k.shape[-1] ** (-0.5), h, *dys, **getattr(self, 'sequence', {}))
            return (*out[:-1], None) if absent_h else out
        finally:
            c.prepare_wy_repr_bwd = oldwy
            c.recompute_w_u_fwd = recompute
            c.chunk_gated_delta_rule_fwd_h = hfn
            c.chunk_bwd_dqkwg = old_direct
