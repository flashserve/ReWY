"""Compute a causal residual Jacobian tile only at its point of consumption."""
import triton
import triton.language as tl


@triton.jit
def residual_dt_tile(p, r, bos, ih, rows, cols, T,
                     HV: tl.constexpr, V: tl.constexpr, VP: tl.constexpr):
    vv = tl.arange(0, VP)
    pp = tl.load(p + ((bos+rows[:, None])*HV+ih)*V+vv[None, :],
                 mask=(rows[:, None]<T)&(vv[None, :]<V),other=0.)
    rr = tl.load(r + ((bos+cols[:, None])*HV+ih)*V+vv[None, :],
                 mask=(cols[:, None]<T)&(vv[None, :]<V),other=0.)
    dd = -tl.dot(pp,tl.trans(rr),allow_tf32=False)
    return tl.where((rows[:,None]>cols[None,:])&(rows[:,None]<T)&(cols[None,:]<T),dd,0.)
