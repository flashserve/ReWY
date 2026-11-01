"""Complete backward bindings. No mutation of upstream module globals."""
from functools import lru_cache
import importlib
import inspect
from types import FunctionType

def _clone(function, overrides):
    original = inspect.unwrap(function)
    result = FunctionType(original.__code__, dict(original.__globals__, **overrides),
                          original.__name__, original.__defaults__, original.__closure__)
    result.__kwdefaults__ = original.__kwdefaults__
    return result

@lru_cache(None)
def _vector(op, device, variant='combined'):
    baseline = importlib.import_module(f'fla.ops.{op}.chunk_bwd')
    if op == 'kda':
        if device == 'cuda':
            from fast_delta_rule.kernels.kda_residual_local_beta import LocalBetaResidual
            from fast_delta_rule.kernels.kda_intra_beta_masked import chunk_kda_bwd_intra
            from fast_delta_rule.kernels.endpoint_intra import EndpointMatrixIntra
            intra = EndpointMatrixIntra(baseline.chunk_kda_bwd_intra, chunk_kda_bwd_intra, True)
            wy = _clone(baseline.chunk_kda_bwd_wy_dqkg_fused,
                        dict(chunk_kda_bwd_kernel_wy_dqkg_fused=LocalBetaResidual()))
            overrides = dict(chunk_kda_bwd_wy_dqkg_fused=wy, chunk_kda_bwd_intra=intra)
        else:
            from fast_delta_rule.kernels import kda_residual_npu_onchip as residual
            if device == 'npu950':
                from fast_delta_rule.kernels.matrix_intra_npu950 import kda_intra as intra
            else:
                from fast_delta_rule.kernels.kda_intra_npu9382 import kda_intra as intra
            residual._CORE_BK = 128
            # Replace the dispatched WY wrapper at the orchestration boundary.
            overrides = dict(chunk_kda_bwd_wy_dqkg_fused=residual.chunk_kda_bwd_wy_dqkg_fused_npu,
                             chunk_kda_bwd_intra=intra)
            if device == 'npu950' and variant == 'residual':
                overrides.pop('chunk_kda_bwd_intra')
                intra = baseline.chunk_kda_bwd_intra
            elif device == 'npu950' and variant == 'matrix':
                overrides.pop('chunk_kda_bwd_wy_dqkg_fused')
        result = _clone(baseline.chunk_kda_bwd, overrides)
        result._fdr_intra = intra
        return result
    if device == 'cuda':
        from fast_delta_rule.kernels.gdn2_consumer_fp32 import residual, matrix_intra
        from fast_delta_rule.kernels.vector_state_gpu import state_bwd
        overrides = dict(chunk_gdn2_bwd_wy_dqkg_fused=residual,
                         chunk_gdn2_bwd_intra=matrix_intra,
                         chunk_gated_delta_rule_bwd_dhu=state_bwd)
    else:
        from fast_delta_rule.kernels.gdn2_full_npu import residual
        if device == 'npu950':
            from fast_delta_rule.kernels.matrix_intra_npu950 import gdn2_intra as matrix_intra
        else:
            from fast_delta_rule.kernels.gdn2_intra_npu9382 import gdn2_intra as matrix_intra
        overrides = dict(chunk_gdn2_bwd_wy_dqkg_fused=residual, chunk_gdn2_bwd_intra=matrix_intra)
        if device == 'npu950' and variant == 'residual':
            overrides.pop('chunk_gdn2_bwd_intra')
        elif device == 'npu950' and variant == 'matrix':
            overrides.pop('chunk_gdn2_bwd_wy_dqkg_fused')
    return _clone(baseline.chunk_gdn2_bwd, overrides)

def kda_backward(*args, **kwargs):
    from fast_delta_rule.runtime import record
    q = args[0] if args else kwargs['q']
    device, variant = _selection(q)
    implementation = _vector('kda', device, variant)
    previous = getattr(implementation._fdr_intra, 'fallback_calls', 0)
    out = implementation(*args, **kwargs)
    if getattr(implementation._fdr_intra, 'fallback_calls', 0)>previous:
        record('kda', 'kernel_wrapper_fallback')
    record('kda', 'optimized_backward')
    return out

def gdn2_backward(*args, **kwargs):
    from fast_delta_rule.runtime import record
    q = args[0] if args else kwargs['q']
    device, variant = _selection(q)
    out = _vector('gdn2', device, variant)(*args, **kwargs)
    record('gdn2', 'optimized_backward')
    return out

def _backend(q):
    if q.device.type == 'npu':
        from .npu_target import device_backend
        return device_backend(q.device)
    return q.device.type

def _selection(q):
    device = _backend(q)
    if device == 'npu950':
        from .npu_target import variant
        return device, variant()
    return device, 'combined'

def _scalar(op, args, kwargs):
    from fast_delta_rule.runtime import record
    module = importlib.import_module('fla.ops.' + ('gated_delta_rule' if op == 'gdn' else 'delta_rule') + '.chunk')
    name = 'chunk_gated_delta_rule_bwd' if op == 'gdn' else 'chunk_delta_rule_bwd'
    bound = inspect.signature(getattr(module, name)).bind(*args, **kwargs)
    bound.apply_defaults()
    a = bound.arguments
    q,k,v,beta = (a[n] for n in ('q','k','v','beta'))
    if _selection(q)[1] != 'combined':
        raise ValueError('950 scalar operators expose the combined adjoint only; matrix/residual ablations are KDA/GDN2 only')
    g = a.get('g')
    h0 = a['initial_state']
    svf = a.get('state_v_first', False)
    common = dict(cu_seqlens=a.get('cu_seqlens'), chunk_indices=a.get('chunk_indices'))
    recompute = module.recompute_w_u_fwd
    if q.device.type == 'npu':
        from fast_delta_rule.kernels.gdn_original_separate_npu import recompute_w_u_fwd_npu
        recompute = recompute_w_u_fwd_npu
    kw = dict(k=k, v=v, beta=beta, A=a['A'], **common)
    if op == 'gdn' or q.device.type == 'npu': kw['g'] = g
    w,u = recompute(**kw)
    h,r,_ = module.chunk_gated_delta_rule_fwd_h(k=k,w=w,u=u,g=g,initial_state=h0,
        output_final_state=False,state_v_first=svf,**common)
    dv = module.chunk_bwd_dv_local(q=q,k=k,g=g,do=a['do'],scale=a['scale'],**common)
    dh,dh0,dr = module.chunk_gated_delta_rule_bwd_dhu(q=q,k=k,w=w,g=g,h0=h0,dht=a['dht'],
        do=a['do'],dv=dv,scale=a['scale'],state_v_first=svf,**common)
    if q.device.type == 'cuda':
        from fast_delta_rule.kernels.scalar_liveness_gpu import adjoint
    elif op == 'gdn':
        from fast_delta_rule.kernels.gdn130_wide_npu import adjoint
    else:
        from fast_delta_rule.kernels.scalar_adjoint_npu import adjoint
    dq,dk,dv,dg,db = adjoint(q,k,v,g,beta,a['A'],dr,r,h,dh,a['do'],a['scale'],state_v_first=svf,**common)
    record(op, 'optimized_backward')
    if op == 'gdn': return dq,dk,dv,db,dg,dh0,None,None
    return dq,dk,dv,db,dh0

def gdn_backward(*args, **kwargs):
    return _scalar('gdn', args, kwargs)

def delta_backward(*args, **kwargs):
    return _scalar('delta', args, kwargs)
