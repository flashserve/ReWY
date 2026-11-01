#!/usr/bin/env python3
"""Explicit 950 acceptance runner. `plan` is CPU-only; nothing runs on import."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback
import warnings

ROOT = Path(__file__).resolve().parents[1]
PREPARED_REVISION = 'npu950-fp32-intra-c32k64-20261029'
CASES = [(1, 512, 4), (1, 2048, 4), (1, 8192, 4),
         (1, 2048, 16), (2, 2048, 16), (1, 8192, 16)]


def plan(ops, variant):
    if variant != 'combined' and any(op not in ('kda', 'gdn2') for op in ops):
        raise ValueError('Separate residual/matrix ablations are available for KDA/GDN2 only')
    return dict(target='physical Ascend950 series', status='awaiting_device_acceptance',
                prepared_revision=PREPARED_REVISION, matrix_schedule='C32/K64; valid-lane loads and explicit masks',
                ops=ops, variant=variant, shapes_B_T_H=CASES, K=128, V=128, chunk_size=64,
                boundaries=['B', 'FB'], timing='public autograd, synchronized eager wall-clock',
                policies={op: ['recompute', 'all'] if op in ('kda', 'gdn2') else ['recompute'] for op in ops},
                precision='frozen input/output/accumulator/storage contract; FP32 intra coefficients',
                rounds=30, comparison='frozen FLA original on the same physical device',
                scope='operator only; public boundary differs from historical low-level graph timing')


def source_hashes():
    files = sorted((ROOT / 'src' / 'fast_delta_rule').rglob('*.py'))
    files += [ROOT / 'src/fast_delta_rule/baseline.json', Path(__file__).resolve(),
              ROOT / 'scripts/npu950_intra_regression.py']
    files += sorted((ROOT / 'validation/npu950_intra').glob('*'))
    return {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def check_prepared(manifest_path):
    """CPU-only identity check; never imports torch or touches a device."""
    expected=json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if expected.get('revision')!=PREPARED_REVISION or expected.get('source_hashes')!=source_hashes():
        raise RuntimeError('Prepared 950 source mismatch: sync the new sealed source bundle; do not run old code')
    return {'revision':PREPARED_REVISION,'source_files':len(expected['source_hashes']),'verified':True}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n', encoding='utf-8')


def preflight(index):
    # Do not query CUDA or auto-select another accelerator.
    import torch
    import torch_npu
    import triton
    from triton.runtime import driver
    from fast_delta_rule.backends.npu_target import require_950
    from fast_delta_rule.runtime import check_baseline
    from fast_delta_rule.kernels import matrix_intra_npu950 as matrix
    if matrix.SCHEDULE_REVISION != PREPARED_REVISION or not Path(matrix.__file__).resolve().is_relative_to(ROOT / 'src'):
        raise RuntimeError('950 runner imported an old/different installed implementation')
    if not torch.npu.is_available():
        raise RuntimeError('No NPU available; no fallback device will be used')
    torch.npu.set_device(index)
    require_950(f'npu:{index}')
    target = driver.active.get_current_target()
    if target.backend != 'npu':
        raise RuntimeError(f'Triton must target NPU, got {target}')
    props = driver.active.utils.get_device_properties(index)
    from fla.utils.ascend_ub_manager import get_npu_properties, UBManager
    fla_props = get_npu_properties()
    with warnings.catch_warnings(record=True) as ub_warnings:
        warnings.simplefilter('always')
        ub = UBManager()
    if any('fallback UB capacity' in str(w.message) for w in ub_warnings):
        raise RuntimeError('Actual UB capacity query failed; fallback capacity is not accepted')
    if ub.ub_capacity_bytes <= 1:
        raise RuntimeError('Invalid actual UB capacity')
    for key in ('num_aicore', 'num_vectorcore'):
        if not isinstance(props.get(key), int) or props[key] <= 0:
            raise RuntimeError(f'Missing real hardware property: {key}')
    # A5 compiler evidence: frozen GDN WY BK/BV128 needs 361216 bytes,
    # exceeding its 253952-byte available UB. One hardware-wide cap for every
    # workload; preserve equations, dtype, reductions and the frozen disk files.
    check_baseline()
    import importlib
    scalar_wy_schedule = {}
    for module_name in ('fla.ops.gated_delta_rule.backends.triton_ascend.wy_fast',
                        'fast_delta_rule.kernels.gdn_original_separate_npu'):
        module = importlib.import_module(module_name)
        old = {key: getattr(module, key) for key in ('_MAX_TILE_BWD_KV', '_MAX_TILE_BWD')}
        module._MAX_TILE_BWD_KV = 64
        module._MAX_TILE_BWD = 64
        scalar_wy_schedule[module_name] = dict(original_caps=old, cap=64,
            reason='Ascend950DT UB compile overflow at BK/BV128; uniform hardware cap')
    gdn2_wy = importlib.import_module('fla.ops.gdn2.backends.triton_ascend.chunk_bwd')
    old_gdn2_cap = gdn2_wy._MAX_TILE
    gdn2_wy._MAX_TILE = 64
    scalar_wy_schedule[gdn2_wy.__name__] = dict(original_cap=old_gdn2_cap, cap=64,
        reason='Ascend950DT original WY gate part requires 319488 bytes at 128; UB is 253952 bytes')
    from fast_delta_rule.kernels.kda_wy_finalize_npu950 import install as install_kda_finalize
    kda_finalize_schedule = install_kda_finalize()
    from fast_delta_rule.kernels.gdn2_wy_finalize_npu950 import install as install_gdn2_finalize
    gdn2_finalize_schedule = install_gdn2_finalize()
    from fast_delta_rule.kernels.scalar_producer_npu950 import install as install_scalar_producer
    scalar_producer_schedule = install_scalar_producer()
    versions = {}
    for name in ('torch', 'torch-npu', 'triton', 'triton-ascend'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cann = os.environ.get('ASCEND_HOME_PATH')
    version_files = {}
    if cann:
        for path in sorted(Path(cann).glob('*version*.cfg')) + sorted(Path(cann).glob('version.info')):
            if path.is_file():
                version_files[path.name] = path.read_text(errors='replace')
    return dict(device_index=index, physical_device=torch.npu.get_device_name(index),
                triton_target=str(target), properties=props, fla_properties=fla_props, versions=versions,
                ub_capacity_bytes=ub.ub_capacity_bytes, ub_model=ub.npu_model,
                ub_capacity_source='CANN get_soc_spec via FLA UBManager; no fallback',
                test_data_root=os.environ.get('TEST_DATA_ROOT_PATH'),
                cann_version_files=version_files, baseline=check_baseline(),
                source_hashes=source_hashes(), native_dav2201_library_used=False, scalar_wy_schedule=scalar_wy_schedule,
                kda_finalize_schedule=kda_finalize_schedule,
                gdn2_finalize_schedule=gdn2_finalize_schedule,
                scalar_producer_schedule=scalar_producer_schedule,
                prepared_revision=matrix.SCHEDULE_REVISION, matrix_module=matrix.__file__)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'check-source', 'preflight', 'verify', 'benchmark'])
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing evidence is never overwritten')
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('--ops', default='kda,gdn,gdn2,delta')
    parser.add_argument('--variant', choices=['residual', 'matrix', 'combined'], default='combined')
    parser.add_argument('--expected-manifest', type=Path, help='Required source seal for device actions and check-source')
    args = parser.parse_args(argv)
    ops = args.ops.split(',')
    if not ops or len(set(ops)) != len(ops) or set(ops) - {'kda', 'gdn', 'gdn2', 'delta'}:
        parser.error('Specify unique supported operators')
    protocol = plan(ops, args.variant)
    if args.action != 'plan' and args.expected_manifest is None:
        parser.error('--expected-manifest is required to prevent running stale prepared code')
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / 'protocol.json', protocol)
    write(args.output / 'source_hashes.json', source_hashes())
    if args.action == 'plan':
        print('Plan saved; no torch import, device query or accelerator execution.')
        return
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['FDR_EXPERIMENTAL_NPU'] = '1'
    os.environ['FDR_NPU950_VARIANT'] = args.variant
    try:
        write(args.output / 'prepared-source-check.json', check_prepared(args.expected_manifest))
        if args.action == 'check-source':
            print('Prepared source hashes verified; no torch import or device execution.')
            return
        test_data=args.output.resolve() / 'tvm-test-data'
        test_data.mkdir(exist_ok=True)
        os.environ.setdefault('TEST_DATA_ROOT_PATH',str(test_data))
        env = preflight(args.device_index)
        write(args.output / 'environment.json', env)
        if args.action == 'preflight':
            return
        from npu950_intra_regression import run as intra_regression
        intra_acceptance=intra_regression(ops,args.output / 'intra-regression.json')
        if not intra_acceptance['passed']:
            raise RuntimeError('Real-input matrix intra regression failed; timing is disabled')
        from fast_delta_rule.validation import verify
        acceptance = verify('npu', ops)
        write(args.output / 'verification.json', acceptance)
        if not acceptance['passed']:
            raise RuntimeError('CPU FP64 acceptance failed; timing is disabled')
        if args.action == 'verify':
            return
        from fast_delta_rule.benchmark import benchmark
        rows = []
        for op in ops:
            for policy in protocol['policies'][op]:
                for boundary in protocol['boundaries']:
                    ratios = []
                    for i, (b, t, h) in enumerate(CASES):
                        result = benchmark('npu', op, t, h, protocol['rounds'], boundary, policy, b)
                        result.update(variant=args.variant, physical_device=env['physical_device'])
                        if result['dispatch'].get(f'{op}.optimized_backward', 0) <= 0:
                            raise RuntimeError('No optimized backward executed')
                        write(args.output / f'{op}_{policy}_{boundary}_{i}.json', result)
                        ratio = result['speedup']
                        if not math.isfinite(ratio) or ratio <= 0:
                            raise RuntimeError('Nonfinite/nonpositive timing ratio')
                        ratios.append(ratio)
                    rows.append(dict(op=op, policy=policy, boundary=boundary, variant=args.variant,
                                     geometric_mean=math.exp(statistics.mean(map(math.log, ratios))),
                                     cases=len(CASES), physical_device=env['physical_device'],
                                     timing=protocol['timing']))
        if env['source_hashes'] != source_hashes():
            raise RuntimeError('Sources changed during measurement; no valid summary')
        write(args.output / 'summary.json', rows)
    except Exception as exc:
        write(args.output / 'failure.json', dict(error=str(exc), traceback=traceback.format_exc(),
                                               status='failed; no complete performance claim'))
        raise


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'src'))
    main()
