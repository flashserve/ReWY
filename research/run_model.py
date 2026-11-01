"""Print or run one measured full-model configuration from the manuscript."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true', help='Execute on the explicitly configured remote accelerator.')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'research/model_runs.json').read_text(encoding='utf-8'))
    rows = manifest['a800'] + manifest['npu9382']
    if args.list:
        for row in rows:
            print(f"{row['case']}: {row['device']}; {row['scope']}; {row['source_folder']}")
        return
    row = next((r for r in rows if r['case'] == args.case), None)
    if row is None:
        parser.error('Select a recorded --case; use --list to see the available cases.')
    directory = ROOT / 'research/models' / row['source_folder']
    command = [sys.executable, str(directory / row['entry']), *row['argv']]
    print(json.dumps(dict(case=row['case'], device=row['device'], scope=row['scope'],
                         timing=row['timing'], command=command), indent=2))
    if not args.run:
        return
    if args.output is None:
        parser.error('--run requires a new --output directory.')
    output = args.output.resolve()
    if output.exists():
        parser.error('Output already exists; preserve previous measurements.')
    if row['device'].startswith('A800') and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        parser.error('Set CUDA_VISIBLE_DEVICES explicitly in the authorized A800 environment.')
    if row['device'] == 'Ascend910_9382' and not os.environ.get('ASCEND_RT_VISIBLE_DEVICES'):
        parser.error('Set ASCEND_RT_VISIBLE_DEVICES explicitly on the 9382 host.')
    output.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, FDR_MODEL_OUTPUT=str(output), KDA_DEVICE='cuda',
               PYTHONPATH=os.pathsep.join([str(ROOT / 'src'), os.environ.get('PYTHONPATH', '')]))
    if row['device'] == 'Ascend910_9382':
        env['FDR_EXPERIMENTAL_NPU'] = '1'
        env['KDA_DEVICE'] = 'npu'
    subprocess.run(command, cwd=directory, env=env, check=True)


if __name__ == '__main__':
    main()
