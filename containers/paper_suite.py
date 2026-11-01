#!/usr/bin/env python3
"""One-shot, platform-specific execution of the current manuscript experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
POLICIES = ['recompute', 'all']


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)


def plan(platform, suite='all'):
    jobs = []
    def operator(group, folder, op, variants, mode, candidate, component=None):
        entry = {'bench': 'run.py', 'correctness': 'run.py', 'generality': 'generality_strict.py'}.get(mode, mode + '.py')
        if mode == 'bench' and platform == 'a800':
            entry = 'delta_bench_stable.py'
        argv = ['--op', op, '--variants', ','.join(variants)]
        if mode in ('bench', 'correctness'):
            argv += ['--mode', mode]
        if mode == 'bench':
            argv += ['--rounds', '30' if platform == 'a800' else '10', '--boundaries', 'B']
        if mode == 'audit':
            argv += ['--skip-stage']
        jobs.append(dict(id=f'{group}-{op}-{mode}', kind='operator', group=group, operator=op,
                         entry=f'research/operators/{folder}/{entry}', argv=argv, mode=mode,
                         variants=variants, candidate=candidate, component=component,
                         boundary='low-level complete B', policies=POLICIES))
    if suite in ('all', 'operators'):
        if platform == 'a800':
            for op, candidate in [('delta', 'native_v16s3'), ('gdn', 'liveness')]:
                for mode in ('correctness', 'extended', 'generality', 'acceptance', 'bench'):
                    operator('main', 'a800_scalar', op, ['original', candidate], mode, candidate)
            for op in ('kda', 'gdn2'):
                for mode in ('correctness', 'extended', 'audit', 'memory_check', 'bench'):
                    operator('main', 'a800_ablation', op, ['original', 'residual', 'matrix', 'combined'], mode,
                             'combined', ['residual', 'matrix', 'combined'])
        else:
            definitions = [('delta', ['original', 'unified'], 'unified', None),
                           ('gdn', ['original', 'wide'], 'wide', None),
                           ('kda', ['original', 'residual', 'matrix', 'combined'], 'combined', ['residual', 'matrix', 'combined']),
                           ('gdn2', ['original', 'matrix', 'full', 'full_combined'], 'full_combined', ['full', 'matrix', 'full_combined'])]
            for op, variants, candidate, component in definitions:
                for mode in ('correctness', 'extended', 'audit', 'bench'):
                    operator('main', 'family_completion_20261024', op, variants, mode, candidate, component)
            # Historical component timing uses its own paired implementation.
            for op, variants, components in [
                ('gdn', ['original', 'gram', 'staged', 'staged_gram'], ['staged_gram', 'gram', 'staged']),
                ('gdn2', ['original', 'residual', 'matrix', 'combined'], ['residual', 'matrix', 'combined'])]:
                for mode in ('correctness', 'extended', 'audit', 'bench'):
                    operator('components', 'family_completion_20261024', op, variants, mode, components[2], components)
    if suite in ('all', 'models'):
        rows = load(ROOT / 'research/model_runs.json')['a800' if platform == 'a800' else 'npu9382']
        for row in rows:
            jobs.append(dict(id='model-' + row['case'], kind='model', case=row['case'], scope=row['scope'],
                             entry='research/models/' + row['source_folder'] + '/' + row['entry'],
                             argv=row['argv'], boundary=row['timing']))
    return dict(platform=platform, suite=suite, jobs=jobs, job_count=len(jobs),
                scope='Current paper only; changed-structure model cases remain separately labelled.',
                precision='Historical matrix controls retain dagger status. Model precision differences are diagnostics, not convergence acceptance.')


def identity():
    paths = []
    for folder in ('src', 'research', 'benchmarks', 'containers', 'scripts', 'validation'):
        paths += [
            p.relative_to(ROOT).as_posix() for p in (ROOT / folder).rglob('*')
            if p.is_file() and p.suffix in ('.py', '.json', '.npz', '.asc', '.h', '.cpp', '.sh', '.txt')
            and not {'__pycache__', 'results', 'build'} & set(p.parts)
        ]
    paths += ['LICENSE', 'THIRD_PARTY_NOTICES.md']
    return {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in sorted(set(paths))}


def verify():
    subprocess.run([sys.executable, str(ROOT / 'scripts/check_reproduction.py')], check=True)
    for platform in ('a800', 'ascend9382'):
        for job in plan(platform)['jobs']:
            if not (ROOT / job['entry']).is_file():
                raise RuntimeError('Missing experiment entry: ' + job['entry'])
    print('PASS: platform plans and complete artifact; no accelerator initialized.', flush=True)


def preflight(platform):
    import torch
    import triton
    from fast_delta_rule.runtime import check_baseline
    result = dict(torch=torch.__version__, triton=triton.__version__, python=sys.version,
                  frozen_fla=check_baseline(), platform=platform)
    if platform == 'a800':
        if not os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise RuntimeError('Set CUDA_VISIBLE_DEVICES to the explicitly selected A800 device.')
        if torch.cuda.device_count() != 1:
            raise RuntimeError('Expose exactly one idle A800 GPU per experiment process.')
        name = torch.cuda.get_device_name(0)
        if 'A800' not in name or torch.cuda.get_device_capability(0) != (8, 0):
            raise RuntimeError('This image protocol requires an A800, observed ' + name)
        if torch.__version__ != '2.9.1+cu128' or triton.__version__ != '3.5.1':
            raise RuntimeError('A800 protocol requires the recorded Torch 2.9.1+cu128 / Triton 3.5.1 environment.')
        result.update(device=name, visible=os.environ['CUDA_VISIBLE_DEVICES'], memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    else:
        import torch_npu
        if not os.environ.get('ASCEND_RT_VISIBLE_DEVICES'):
            raise RuntimeError('Set ASCEND_RT_VISIBLE_DEVICES to one explicitly selected 9382 device.')
        torch.npu.set_device(0)
        name = torch.npu.get_device_name(0)
        if name != 'Ascend910_9382' or torch.npu.device_count() != 1:
            raise RuntimeError('Exactly one Ascend910_9382 is required, observed ' + name)
        if torch.__version__ != '2.7.1+cpu' or torch_npu.__version__ != '2.7.1.post4':
            raise RuntimeError('9382 protocol requires the recorded Torch / TorchNPU pair.')
        result.update(device=name, torch_npu=torch_npu.__version__, visible=os.environ['ASCEND_RT_VISIBLE_DEVICES'])
    return result


def snapshot(output, hashes):
    dest = output / 'source'
    for name, expected in hashes.items():
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Source changed while snapshotting: ' + name)
    native = Path(os.environ.get('FDR_PAPER_NATIVE_LIBRARY', '/opt/fdr-native/libkda_residual_native.so'))
    if native.exists():
        target = dest / 'research/operators/cannbot_npu_20261024/native/build' / native.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(native, target)
        write(output / 'native-library.json', dict(sha256=hashlib.sha256(native.read_bytes()).hexdigest(), source=str(native)))
    return dest


def replay(output):
    output.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / 'benchmarks/plotting/make_figures.py'), '--output-dir', str(output / 'figures')], check=True)
    for name in ('paper_data.json', 'e2e_data.json', 'npu950_data.json', 'h100_data.json',
                 'h200_data.json', 'presentation_scope.json'):
        shutil.copyfile(ROOT / 'benchmarks/data' / name, output / name)
    for name in ('figure_data.json', 'intro_graph_data.json'):
        if load(output / 'figures' / name) != load(ROOT / 'benchmarks/data' / name):
            raise RuntimeError('Replayed numerical figure values differ: ' + name)
    write(output / 'replay.json', dict(mode='published-aggregate-replay', new_measurements=False,
                                        source_sha256=identity(), figures='All six original paper figures; numerical values checked.'))
    print('PASS: archived paper replay saved to ' + str(output), flush=True)


def run(args):
    output = args.output.resolve()
    protocol = plan(args.platform, args.suite)
    if args.only:
        wanted = set(args.only.split(','))
        known = {j['id'] for j in protocol['jobs']}
        if not wanted <= known:
            raise ValueError('Unknown jobs: ' + str(wanted - known))
        protocol['jobs'] = [j for j in protocol['jobs'] if j['id'] in wanted]
        protocol['job_count'] = len(protocol['jobs'])
        protocol['partial_selection'] = True
    hashes = identity()
    if args.resume:
        state = load(output / 'state.json')
        if load(output / 'source-hashes.json') != hashes or load(output / 'protocol.json') != protocol:
            raise RuntimeError('Cannot resume with changed sources, platform or job selection.')
        source = output / 'source'
    else:
        output.mkdir(parents=True, exist_ok=False)
        write(output / 'protocol.json', protocol)
        write(output / 'source-hashes.json', hashes)
        source = snapshot(output, hashes)
        state = dict(status='preflight', jobs={}, started=time.time())
    write(output / 'state.json', state)
    write(output / 'environment.json', preflight(args.platform))
    env = dict(os.environ, KDA_DEVICE='cuda' if args.platform == 'a800' else 'npu',
               PYTHONPATH=os.pathsep.join([str(source / 'src'), os.environ.get('PYTHONPATH', '')]),
               FDR_EXPERIMENTAL_NPU='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    for job in protocol['jobs']:
        if state['jobs'].get(job['id'], {}).get('status') == 'complete':
            continue
        directory = output / 'jobs' / job['id']
        if directory.exists():
            # Retain failed/partial output; retry in a separately numbered attempt.
            number = 1
            while directory.with_name(directory.name + f'.attempt{number}').exists():
                number += 1
            directory = directory.with_name(directory.name + f'.attempt{number}')
        directory.mkdir(parents=True)
        entry = source / job['entry']
        command = [sys.executable, '-u', str(entry), *job['argv']]
        run_env = dict(env)
        if job['kind'] == 'operator':
            command += ['--output', str(directory / 'records.jsonl')]
        else:
            run_env['FDR_MODEL_OUTPUT'] = str(directory / 'model')
        state.update(status='running', current=job['id'])
        state['jobs'][job['id']] = dict(status='running', directory=directory.relative_to(output).as_posix(), started=time.time())
        write(output / 'state.json', state)
        write(directory / 'command.json', dict(argv=command, boundary=job['boundary']))
        print('START ' + job['id'], flush=True)
        with (directory / 'run.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=entry.parent, env=run_env, stdout=log, stderr=subprocess.STDOUT)
        record = state['jobs'][job['id']]
        record.update(status='complete' if result.returncode == 0 else 'failed', returncode=result.returncode, finished=time.time())
        write(output / 'state.json', state)
        print(record['status'].upper() + ' ' + job['id'], flush=True)
        if result.returncode:
            state['status'] = 'failed'
            write(output / 'state.json', state)
            raise RuntimeError('Experiment failed; inspect ' + str(directory / 'run.log'))
    state.update(status='complete', current=None, finished=time.time())
    write(output / 'state.json', state)
    subprocess.run([sys.executable, str(ROOT / 'containers/summarize.py'), str(output)], check=True)
    # A single run delivers new platform measurements and the original paper
    # panels in separately labelled directories; never mix the two datasets.
    rendered = output / 'paper-replay'
    attempt = 0
    while rendered.exists():
        provenance = rendered / 'replay.json'
        if provenance.is_file() and load(provenance).get('source_sha256') == hashes:
            break
        attempt += 1
        rendered = output / f'paper-replay.attempt{attempt}'
    else:
        replay(rendered)
    state['paper_replay'] = rendered.relative_to(output).as_posix()
    write(output / 'state.json', state)
    print('Completed experiment data and figures: ' + str(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan', 'verify', 'replay', 'run', 'summarize'])
    parser.add_argument('--platform', choices=['a800', 'ascend9382'], default=os.environ.get('FDR_PAPER_PLATFORM'))
    parser.add_argument('--suite', choices=['all', 'operators', 'models'], default='all')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--only', help='Comma-separated job IDs for explicit partial validation; never labelled a full reproduction.')
    args = parser.parse_args()
    if args.action == 'verify':
        verify()
        return
    if args.action in ('run', 'plan') and not args.platform:
        parser.error('--platform is required outside a platform image.')
    if args.action == 'plan':
        print(json.dumps(plan(args.platform, args.suite), indent=2))
        return
    if not args.output:
        parser.error('--output is required; choose a persistent mounted destination.')
    if args.action == 'replay':
        replay(args.output.resolve())
    elif args.action == 'summarize':
        subprocess.run([sys.executable, str(ROOT / 'containers/summarize.py'), str(args.output.resolve())], check=True)
    else:
        run(args)


if __name__ == '__main__':
    main()
