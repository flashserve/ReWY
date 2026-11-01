import importlib
from common import *
from gdn2_kernels_gpu import residual, matrix_intra
f = importlib.import_module('fla.ops.gdn2.chunk_fwd')
c = importlib.import_module('fla.ops.gdn2.chunk_bwd')

class GDN2:
    name = 'gdn2'
    names = ['dq', 'dk', 'dv', 'dg', 'db', 'dw', 'dh0']
    variants = ['original', 'residual', 'matrix', 'combined']

    def gate_diagnostics(self, fw):
        g = fw[0][2].detach().float().cpu()
        spans = []
        for t in range(0, g.shape[1], 64):
            z = g[:, t:t + 64]
            spread = z.amax(1) - z.amin(1)
            for k in range(0, g.shape[-1], 32):
                spans.extend(spread[..., k:k + 32].amax(-1).flatten().tolist())
        return dict(tiles=len(spans), span_min=min(spans), span_max=max(spans), matrix_tiles_at_16=sum((s < 16 for s in spans)), matrix_tiles_at_80=sum((s < 80 for s in spans)))

    def inputs(self, case):
        kw = {k: v for k, v in case.items() if k not in ('zero_beta', 'tiny_beta', 'positive_gate')}
        kw.pop('HV', None)
        xs, dys = e.inputs(**kw)
        q, k, v, g, beta, h = xs
        b = (torch.rand_like(k) * 0.8 + 0.1).detach().requires_grad_()
        w = (torch.rand_like(v) * 0.8 + 0.1).detach().requires_grad_()
        with torch.no_grad():
            if case.get('zero_beta'):
                b[:, ::3] = 0
                w[:, 1::3] = 0
            if case.get('tiny_beta'):
                b[:, ::3] = 1e-09
                w[:, 1::3] = 1e-09
            if case.get('positive_gate'):
                g[:, ::3].mul_(-1)
        return ((q, k, v, g, b, w, h), dys)

    def oracle(self, xs, svf=False):
        q, k, v, g, b, w, h = xs
        if svf:
            h = h.transpose(-1, -2)
        outputs = []
        for t in range(q.shape[1]):
            h = h * g[:, t].exp()[..., None]
            r = w[:, t] * v[:, t] - torch.einsum('bhk,bhkv->bhv', b[:, t] * k[:, t], h)
            h = h + k[:, t, ..., None] * r[..., None, :]
            outputs.append(torch.einsum('bhk,bhkv->bhv', q[:, t] * q.shape[-1] ** (-0.5), h))
        return (torch.stack(outputs, 1), h.transpose(-1, -2) if svf else h)

    def forward(self, xs, policy, svf=False):
        q, k, v, g, b, w, h = xs
        z = f.chunk_gdn2_fwd(q, k, v, g, b, w, k.shape[-1] ** (-0.5), h, True, disable_recompute=policy == 'all', state_v_first=svf, **getattr(self, 'sequence', {}))
        return (z, policy, z[:2])

    def backward(self, xs, dys, fw, variant, svf=False):
        z, policy, _ = fw
        q, k, v, g, b, w, h = xs
        wy, intra = (c.chunk_gdn2_bwd_wy_dqkg_fused, c.chunk_gdn2_bwd_intra)
        try:
            if variant in ('residual', 'combined'):
                c.chunk_gdn2_bwd_wy_dqkg_fused = residual
            if variant in ('matrix', 'combined'):
                c.chunk_gdn2_bwd_intra = matrix_intra
            if getattr(self, 'record_stage', False):
                stage, intrastage = (c.chunk_gdn2_bwd_wy_dqkg_fused, c.chunk_gdn2_bwd_intra)
                stash = {}

                def recwy(**kw):
                    stash['wy'] = kw
                    return stage(**kw)

                def recintra(**kw):

                    def call():
                        qd, kd, vd, bd, wd, gd, md = stage(**stash['wy'])
                        args = dict(kw, dq=qd, dk=kd, db=bd, dg=gd, dAkk=md)
                        qd, kd, bd, gd = intrastage(**args)
                        return (qd, kd, vd, bd, wd, gd)
                    self.last_stage = call
                    return intrastage(**kw)
                c.chunk_gdn2_bwd_wy_dqkg_fused = recwy
                c.chunk_gdn2_bwd_intra = recintra
            dq, dk, dv, db, dw, dg, dh, *_ = c.chunk_gdn2_bwd(q, k, v, b, w, z[3], z[4], k.shape[-1] ** (-0.5), h, *dys, g=z[2], state_v_first=svf, w_wy=z[5], u_wy=z[6], qg=z[7], kg=z[8], v_new=z[9], h=z[10], disable_recompute=policy == 'all', **getattr(self, 'sequence', {}))
            return (dq, dk, dv, dg, db, dw, dh)
        finally:
            c.chunk_gdn2_bwd_wy_dqkg_fused = wy
            c.chunk_gdn2_bwd_intra = intra
