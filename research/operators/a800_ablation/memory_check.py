"""Three repeated peak-allocation measurements with matching gradient checks."""
import argparse
import gc
from pathlib import Path
import sys
from common import *
from audit import saved_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--op', required=True)
    parser.add_argument('--variants', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.op == 'kda':
        from kda_op import KDA
        op = KDA()
    elif args.op == 'gdn2':
        from gdn2_op import GDN2
        op = GDN2()
    else:
        from scalar_ops import ScalarOp
        op = ScalarOp(args.op)
    path = HERE / 'results' / args.output
    assert not path.exists()
    variants = args.variants.split(',')
    emit({'type': 'environment', 'boundary': 'warmed complete-backward incremental allocated peak, including returned gradients',
          'device': torch.cuda.get_device_name(), 'args': vars(args), 'hashes': hashes()}, path)
    for T, H in ((2048, 4), (8192, 16)):
        case = dict(B=1, T=T, H=H, D=128)
        xs, dys = op.inputs(case)
        for policy in ('recompute', 'all'):
            fw = op.forward(xs, policy)
            ref = tuple(x.cpu().double() for x in op.backward(xs, dys, fw, 'original'))
            ACC.synchronize()
            for repeat in range(3):
                for variant in variants if repeat % 2 == 0 else variants[::-1]:
                    warm = op.backward(xs, dys, fw, variant)
                    ACC.synchronize()
                    del warm
                    gc.collect()
                    before = ACC.memory_allocated()
                    ACC.reset_peak_memory_stats()
                    out = op.backward(xs, dys, fw, variant)
                    ACC.synchronize()
                    peak = ACC.max_memory_allocated() - before
                    values = metrics(ref, out, op.names)
                    ok = check(values)
                    emit(dict(type='memory', case=case, policy=policy, variant=variant,
                              repetition=repeat, backward_peak_extra_bytes=peak,
                              saved_auxiliary_bytes=saved_bytes(fw, xs),
                              gradient_bytes=sum(x.nbytes for x in out),
                              additional_resident_workspace_bytes=0, passed=ok, metrics=values), path)
                    assert ok, (args.op, case, policy, variant, values)
                    del out
            del fw, ref
        del xs, dys
        gc.collect()
        ACC.empty_cache()
    emit(dict(type='dependencies', **dependencies()), path)
    emit(dict(type='complete', passed=True), path)


if __name__ == '__main__':
    with ACC.stream(ACC.Stream()):
        main()
