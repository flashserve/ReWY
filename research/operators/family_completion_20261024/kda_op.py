"""The missing no-DE NPU 2x2 ablation, with the existing frozen implementations."""
import importlib, contextlib, hashlib
from common import *
f = importlib.import_module('fla.ops.kda.chunk_fwd')
c = e.bwd
BASE_INTRA = c.chunk_kda_bwd_intra
if DEV == 'npu':
    from kernels import kda_residual_npu_onchip as residual
    residual._CORE_BK = 128
    sys.path.insert(0, str(ROOT / 'cannbot_npu_20261024'))
    from cannbot_npu_20261024 import op as native
    # Rebuild from the retained native sources; the original binary SHA remains
    # in raw evidence. New compiler/build paths need not give identical ELF bytes.
else:
    from kernels.kda_residual_local_beta import LocalBetaResidual
    from kernels.kda_intra_beta_masked import chunk_kda_bwd_intra as residual_intra
    from gpu_matrix_20261024.matrix_intra import MatrixIntra

class KDA:
    name = 'kda'
    names = ['dq', 'dk', 'dv', 'dg', 'dbeta', 'dh0']
    variants = ['original', 'residual', 'matrix', 'combined']

    def reset_dispatch(self):
        if DEV == 'npu':
            native.dispatch_stats(reset=True)

    def dispatch(self):
        return native.dispatch_stats() if DEV == 'npu' else {}

    def inputs(self, case):
        xs, dys = e.inputs(**{k: v for k, v in case.items() if k not in ('zero_beta', 'tiny_beta', 'positive_gate')})
        with torch.no_grad():
            if case.get('zero_beta'):
                xs[4][:, ::3] = 0
            if case.get('tiny_beta'):
                xs[4][:, ::3] = 1e-09
            if case.get('positive_gate'):
                xs[3][:, ::3].mul_(-1)
        return (xs, dys)

    def oracle(self, xs, svf=False):
        return e.oracle(xs, svf)

    def forward(self, xs, policy, svf=False):
        q, k, v, g, b, h = xs
        z = f.chunk_kda_fwd(q, k, v, g, b, k.shape[-1] ** (-0.5), h, True, state_v_first=svf, disable_recompute=policy == 'all', **getattr(self, 'sequence', {}))
        return (z, policy, z[:2])

    def backward(self, xs, dys, fw, variant, svf=False):
        z, policy, _ = fw
        q, k, v, g, b, h = xs
        isres = variant in ('residual', 'combined')
        ismat = variant in ('matrix', 'combined')
        oldintra = c.chunk_kda_bwd_intra
        if DEV == 'npu':
            oldwy = e.npu_bwd.chunk_kda_bwd_wy_dqkg_fused_npu
            oldnative = (native._old_intra, native._variant)
            if isres:
                e.npu_bwd.chunk_kda_bwd_wy_dqkg_fused_npu = residual.chunk_kda_bwd_wy_dqkg_fused_npu
            if ismat:
                native._old_intra = BASE_INTRA
                native._variant = 'matrix'
                c.chunk_kda_bwd_intra = native.native_intra
        else:
            oldwy = c.chunk_kda_bwd_kernel_wy_dqkg_fused
            if isres:
                c.chunk_kda_bwd_kernel_wy_dqkg_fused = LocalBetaResidual()
                c.chunk_kda_bwd_intra = residual_intra
            if ismat:
                c.chunk_kda_bwd_intra = MatrixIntra(BASE_INTRA, residual_intra, isres, 32, 4, True)
        oldstate = c.chunk_gated_delta_rule_bwd_dhu
        try:
            r = c.chunk_kda_bwd(q, k, v, b, z[3], z[4], k.shape[-1] ** (-0.5), h, *dys, g=z[2], state_v_first=svf, disable_recompute=policy == 'all', w=z[5], u=z[6], qg=z[7], kg=z[8], v_new=z[9], h=z[10], **getattr(self, 'sequence', {}))
            return (r[0], r[1], r[2], r[4], r[3], r[5])
        finally:
            c.chunk_gated_delta_rule_bwd_dhu = oldstate
            c.chunk_kda_bwd_intra = oldintra
            if DEV == 'npu':
                e.npu_bwd.chunk_kda_bwd_wy_dqkg_fused_npu = oldwy
                native._old_intra, native._variant = oldnative
            else:
                c.chunk_kda_bwd_kernel_wy_dqkg_fused = oldwy
