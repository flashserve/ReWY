"""Explicit launcher for the paper's historical low-level operator boundaries."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STUDIES = {
    'rtx-scalar': ('gpu_target120_20261025', 'cuda', {'delta': ['original', 'v16_stream'], 'gdn': ['original', 'v16_stream']}),
    'rtx-vector': ('gpu_vector_followup_20261026', 'cuda', {'kda': ['original', 'combined'], 'gdn2': ['original', 'combined']}),
    'a800-scalar': ('a800_scalar', 'cuda', {'delta': ['original', 'native_v16s3'], 'gdn': ['original', 'liveness']}),
    'a800-vector-historical': ('a800_ablation', 'cuda', {'kda': ['original', 'residual', 'matrix', 'combined'], 'gdn2': ['original', 'residual', 'matrix', 'combined']}),
    '9382-historical': ('family_completion_20261024', 'npu', {'delta': ['original', 'unified'], 'gdn': ['original', 'wide'], 'kda': ['original', 'residual', 'matrix', 'combined'], 'gdn2': ['original', 'matrix', 'full', 'full_combined']}),
    'rtx-components': ('family_completion_20261024', 'cuda', {'kda': ['original', 'residual', 'matrix', 'combined'], 'gdn2': ['original', 'residual', 'matrix', 'combined'], 'gdn': ['original', 'gram', 'gpu_probe', 'gpu_probe_gram']}),
    '9382-components': ('family_completion_20261024', 'npu', {'kda': ['original', 'residual', 'matrix', 'combined'], 'gdn2': ['original', 'matrix', 'full', 'full_combined'], 'gdn': ['original', 'gram', 'staged', 'staged_gram']}),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', required=True, choices=STUDIES)
    p.add_argument('--op', required=True)
    p.add_argument('--mode', choices=['correctness', 'extended', 'bench', 'audit'], default='correctness')
    p.add_argument('--output', type=Path)
    p.add_argument('--run', action='store_true', help='Execute on the study device; default only prints the plan.')
    a = p.parse_args()
    folder, device, operators = STUDIES[a.study]
    if a.op not in operators:
        p.error('This operator is not part of the selected study.')
    variants = operators[a.op]
    entry = 'run.py' if a.mode in ('correctness', 'bench') else a.mode + '.py'
    if a.mode == 'bench' and (a.study.startswith('a800-') or a.study == 'rtx-scalar'):
        entry = 'delta_bench_stable.py'
    command = [sys.executable, str(ROOT / 'operators' / folder / entry), '--op', a.op,
               '--variants', ','.join(variants), '--output', str(a.output.resolve()) if a.output else 'paper-run.jsonl']
    if a.mode in ('correctness', 'bench'):
        command += ['--mode', a.mode]
    if a.mode == 'bench':
        command += ['--rounds', '10' if device == 'npu' else '30', '--boundaries', 'B']
    print(json.dumps(dict(study=a.study, boundary='historical low-level complete B', policies=['recompute', 'all'],
                         device=device, variants=variants, command=command), indent=2))
    if not a.run:
        return
    if a.output is None or a.output.exists():
        p.error('--run requires a new --output JSONL file.')
    if device == 'cuda' and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        p.error('Set CUDA_VISIBLE_DEVICES explicitly on the authorized remote device.')
    if device == 'npu' and not os.environ.get('ASCEND_RT_VISIBLE_DEVICES'):
        p.error('Set ASCEND_RT_VISIBLE_DEVICES explicitly on the 9382 host.')
    a.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(command, env=dict(os.environ, KDA_DEVICE=device), check=True)


if __name__ == '__main__':
    main()
