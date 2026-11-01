"""950 scalar residual producer: same equations/dtypes, one task per program.

Original persistent producer gives allocation-dependent dK/db on long DeltaNet.
Only the producer schedule changes; consumer and residual mathematics are intact.
"""
import triton
import triton.language as tl
from fla.utils.ascend_ub_manager import ASCEND_MAX_GRID_DIM

@triton.jit
def scalar_producer_950(A, DR, R, V0, BETA, DO, P, DV, DBV, DM, DS, CU, CI, T: tl.constexpr, B: tl.constexpr, HV: tl.constexpr, V: tl.constexpr, C: tl.constexpr, NV: tl.constexpr, VAR: tl.constexpr, TASKS: tl.constexpr, STEP: tl.constexpr, TASK_OFFSET: tl.constexpr=0):
    task = tl.program_id(0) + TASK_OFFSET
    it = task // (B * HV)
    bh = task % (B * HV)
    ih = bh % HV
    if VAR:
        seq = tl.load(CI + 2 * it)
        it = tl.load(CI + 2 * it + 1)
        bos = tl.load(CU + seq)
        length = tl.load(CU + seq + 1) - bos
    else:
        bos = bh // HV * T
        length = T
    rr = tl.arange(0, C)
    tt = it * C + rr
    off = (bos + tt) * HV + ih
    soff = bh // HV * HV * T + ih * T + tt if not VAR else ih * T + bos + tt
    beta = tl.load(BETA + soff, tt < length, 0).to(tl.float32)
    dm = tl.zeros((C, C), tl.float32)
    ds = tl.zeros((C, C), tl.float32)
    db = tl.zeros((C,), tl.float32)
    for iv in range(tl.cdiv(V, NV)):
        vv = iv * NV + tl.arange(0, NV)
        vm = (tt[:, None] < length) & (vv[None, :] < V)
        at = tl.load(A + off[None, :] * C + rr[:, None], (tt[:, None] < length) & (tt[None, :] < length), 0)
        dr = tl.load(DR + off[:, None] * V + vv[None, :], vm, 0)
        p = tl.dot(at, dr, allow_tf32=False)
        value = tl.load(V0 + off[:, None] * V + vv[None, :], vm, 0)
        db += tl.sum(p * value, 1)
        tl.store(P + off[:, None] * V + vv[None, :], p, vm)
        tl.store(DV + off[:, None] * V + vv[None, :], p * beta[:, None], vm)
        rt = tl.load(R + off[None, :] * V + vv[:, None], (tt[None, :] < length) & (vv[:, None] < V), 0)
        dm -= tl.dot(p.to(rt.dtype), rt, allow_tf32=False)
        dout = tl.load(DO + off[:, None] * V + vv[None, :], vm, 0)
        rt2 = tl.load(R + off[None, :] * V + vv[:, None], (tt[None, :] < length) & (vv[:, None] < V), 0)
        ds += tl.dot(dout, rt2, allow_tf32=False)
    tl.store(DM + off[:, None] * C + rr[None, :], tl.where(rr[:, None] > rr[None, :], dm, 0), tt[:, None] < length)
    tl.store(DS + off[:, None] * C + rr[None, :], tl.where(rr[:, None] >= rr[None, :], ds, 0), tt[:, None] < length)
    tl.store(DBV + soff, db, tt < length)

class Producer950:
    def __getitem__(self, unused_persistent_grid):
        def launch(**kwargs):
            tasks=kwargs['TASKS']
            for offset in range(0,tasks,ASCEND_MAX_GRID_DIM):
                scalar_producer_950[(min(tasks-offset,ASCEND_MAX_GRID_DIM),)](TASK_OFFSET=offset,**kwargs)
        return launch

def install():
    from fast_delta_rule.backends.npu_target import require_950
    require_950('npu:0')
    from . import scalar_adjoint_npu
    scalar_adjoint_npu.producer=Producer950()
    return dict(id='scalar-residual-producer-one-task-per-program',
                reason='DeltaNet long-sequence candidate repeatability fails in persistent producer; producer-only isolation restores acceptance',
                scope='all 950 scalar-adjoint workloads; original FLA and consumer unchanged',
                precision='original equations, FP32 P/DM/DS/DBV and all operand/output conversions preserved')
