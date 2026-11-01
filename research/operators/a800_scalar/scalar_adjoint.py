"""Preserve the accepted forward/save policy and replace the chunk adjoint."""
from common import DEV

def backward(op, xs, dys, fw, svf=False, variant='unified'):
    c = op.c
    z, cache, _ = fw
    seq = getattr(op, 'sequence', {})
    if op.gated:
        q, k, v, _, beta, h0 = xs
        g = z[0]
        a = z[2]
    else:
        q, k, v, beta, h0 = xs
        g = None
        a = z[1]
    common = dict(cu_seqlens=seq.get('cu_seqlens'), chunk_indices=seq.get('chunk_indices'))
    scale = k.shape[-1] ** (-0.5)
    if 'wu' in cache:
        w, u = cache['wu']
    else:
        args = dict(k=k, v=v, beta=beta, A=a, **common)
        if op.gated:
            args['g'] = g
        w, u = c.recompute_w_u_fwd(**args)
    if 'hr' in cache:
        h, r, _ = cache['hr']
    else:
        h, r, _ = c.chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g, initial_state=h0, output_final_state=False, state_v_first=svf, **common)
    dv = c.chunk_bwd_dv_local(q=q, k=k, g=g, do=dys[0], scale=scale, **common)
    state_bwd = c.chunk_gated_delta_rule_bwd_dhu
    if variant in ('native_v16s3',):
        from functools import partial
        from scalar_state_reference_gpu import state_bwd as reference_state_bwd
        state_bwd = partial(reference_state_bwd, stages=2 if variant == 'native_v16s2' else 3, value_block=16)
    dh, dh0, dr = state_bwd(q=q, k=k, w=w, g=g, h0=h0, dht=dys[1], do=dys[0], dv=dv, scale=scale, state_v_first=svf, **common)
    if variant in ('native_v16s3',):
        from functools import partial
        from scalar_consumer_schedule_gpu import adjoint as schedule_adjoint
        adjoint = partial(schedule_adjoint, nk=16 if variant == 'narrow16' else 32, stages=3 if variant == 'consumer_s3' else 1 if variant == 'narrow16' else 2)
    else:
        from scalar_liveness_gpu import adjoint
    dq, dk, dv, dg, db = adjoint(q, k, v, g, beta, a, dr, r, h, dh, dys[0], scale, state_v_first=svf, **common)
    return (dq, dk, dv, dg, db, dh0) if op.gated else (dq, dk, dv, db, dh0)
