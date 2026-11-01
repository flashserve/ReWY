"""Supplement frozen KDA/GPU component and joint-schedule allocation evidence.

No timing, optimization, or dtype changes. The five implementations are exactly
those in gpu_matrix_20261024's accepted paired timing; verify their hashes first.
Use the existing warmed full-backward incremental allocation boundary. Repeat in
both orders, and keep full-gradient validation outside the measured allocation.
"""
import argparse
import gc
from common import *
from audit import saved_bytes
from kda_op import KDA

sys.path.insert(0, str(ROOT / 'gpu_matrix_20261024'))
from gpu_matrix_20261024.op import implementation, VARIANTS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    assert DEV == 'cuda'
    path = HERE / 'results' / args.output
    if path.exists():
        raise FileExistsError(path)
    repo = ROOT.parents[1]
    import subprocess
    subprocess.run([sys.executable,str(repo/'scripts/check_reproduction.py')],check=True)
    emit(dict(type='environment', device=DEV,
              device_name=torch.cuda.get_device_name(), torch=torch.__version__,
              hashes=hashes(), public_source_check=True,
              boundary='warmed full-backward incremental allocated peak; includes returned gradients',
              precision='unchanged frozen implementations; no new buffers or arithmetic'), path)
    op = KDA()
    for T, H in ((2048, 4), (8192, 16)):
        case = dict(B=1, T=T, H=H, D=128)
        xs, dys = op.inputs(case)
        for policy in ('recompute', 'all'):
            fw = op.forward(xs, policy)
            with implementation('original'):
                ref = op.backward(xs, dys, fw, 'original')
            ACC.synchronize()
            reference = tuple(x.cpu().double() for x in ref)
            del ref
            for repetition in range(3):
                order = VARIANTS if repetition % 2 == 0 else VARIANTS[::-1]
                for variant in order:
                    with implementation(variant) as matrix:
                        tmp = op.backward(xs, dys, fw, 'original')
                        ACC.synchronize()
                        del tmp
                        # Diagnostic output belongs to the next invocation, not
                        # to resident workspace; match the existing audit's fresh
                        # MatrixIntra object per call rather than subtract it.
                        matrix.flags = None
                        gc.collect()
                        before = ACC.memory_allocated()
                        ACC.reset_peak_memory_stats()
                        out = op.backward(xs, dys, fw, 'original')
                        ACC.synchronize()
                        peak = ACC.max_memory_allocated() - before
                        grad_bytes = sum(x.nbytes for x in out)
                        m = metrics(reference, out, op.names)
                        assert check(m), (case, policy, variant, m)
                        del out
                    emit(dict(type='memory', case=case, policy=policy, variant=variant,
                              repetition=repetition, backward_peak_extra_bytes=peak,
                              saved_auxiliary_bytes=saved_bytes(fw, xs),
                              gradient_bytes=grad_bytes, additional_resident_workspace_bytes=0,
                              metrics=m, passed=True), path)
            del fw, reference
        del xs, dys
        gc.collect()
        ACC.empty_cache()
    emit(dict(type='dependencies', **dependencies()), path)
    emit(dict(type='complete', passed=True, timing_rerun=False), path)


if __name__ == '__main__':
    with ACC.stream(ACC.Stream()):
        main()
