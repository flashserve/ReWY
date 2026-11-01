#!/usr/bin/env python3
"""Launch a pinned paper image; the host only needs Python and Docker."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError('Device, card and chip IDs must be nonnegative.')
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('platform', choices=('a800', 'ascend9382'))
    parser.add_argument('--output', required=True, type=Path, help='New directory for persistent results.')
    parser.add_argument('--device', type=nonnegative, help='One idle physical accelerator ID.')
    parser.add_argument('--card', type=nonnegative, help='Ascend npu-smi physical card ID.')
    parser.add_argument('--chip', type=nonnegative, help='Ascend npu-smi chip ID within that card.')
    parser.add_argument('--suite', choices=('all', 'operators', 'models'), default='all')
    parser.add_argument('--figures-only', action='store_true', help='Replay six archived figures on CPU; no device mounts.')
    parser.add_argument('--resume', action='store_true', help='Resume the same experiment directory and suite.')
    parser.add_argument('--plan', action='store_true', help='Print the Docker command; do not run Docker or create files.')
    parser.add_argument('--ascend-root', type=Path, default=Path('/usr/local/Ascend'))
    parser.add_argument('--npu-smi', type=Path, default=Path('/usr/local/sbin/npu-smi'))
    parser.add_argument('--ascend-install-info', type=Path, default=Path('/etc/ascend_install.info'))
    args = parser.parse_args(argv)
    if args.figures_only:
        if args.resume or args.suite != 'all' or any(v is not None for v in (args.device, args.card, args.chip)):
            parser.error('--figures-only uses a new output directory and no device/suite/resume options.')
    elif args.device is None:
        parser.error('Select an idle --device, or use --figures-only for CPU replay.')
    elif args.platform == 'ascend9382' and (args.card is None or args.chip is None):
        parser.error('Ascend experiments also require --card and --chip matching the physical device.')
    elif args.platform == 'a800' and (args.card is not None or args.chip is not None):
        parser.error('--card and --chip are only used on Ascend.')
    args.output = args.output.expanduser().resolve()
    if not args.output.name:
        parser.error('--output must name a result directory, not a filesystem root.')
    return args


def docker_command(args):
    images = json.loads((ROOT / 'containers/images.json').read_text(encoding='utf-8'))
    image = images[args.platform]
    command = ['docker', 'run', '--rm', '--platform', image['platform'], '--network', 'none']
    if not args.figures_only:
        command += ['--ipc=host']
        if args.platform == 'a800':
            command += ['--gpus', f'device={args.device}', '-e', 'CUDA_VISIBLE_DEVICES=0']
        else:
            command += ['--privileged', '--pid=host',
                        '-v', f'{args.ascend_root.as_posix()}:/usr/local/Ascend:ro',
                        '-v', f'{args.npu_smi.as_posix()}:/usr/local/sbin/npu-smi:ro',
                        '-v', f'{args.ascend_install_info.as_posix()}:/etc/ascend_install.info:ro',
                        '-e', f'ASCEND_RT_VISIBLE_DEVICES={args.device}',
                        '-e', f'FDR_NPU_SMI_CARD={args.card}',
                        '-e', f'FDR_NPU_SMI_CHIP={args.chip}']
    # Mount the parent so the in-image runner can create the new output itself.
    command += ['-v', f'{args.output.parent}:/results', image['image'] + '@' + image['digest'],
                'replay' if args.figures_only else 'run', '--output', '/results/' + args.output.name]
    if not args.figures_only:
        command += ['--suite', args.suite]
        if args.resume:
            command += ['--resume']
    return command


def main(argv=None):
    args = parse_args(argv)
    command = docker_command(args)
    print('Docker command (POSIX shell syntax):\n' + shlex.join(command), flush=True)
    print('Results: ' + str(args.output), flush=True)
    if args.plan:
        return 0
    if not shutil.which('docker'):
        raise SystemExit('Docker is not installed/on PATH. See containers/README.md.')
    if args.output.exists() and not args.resume:
        raise SystemExit('Output already exists. Choose a new directory, or --resume the same experiment.')
    if args.resume and not (args.output / 'state.json').is_file():
        raise SystemExit('--resume requires an existing experiment with state.json.')
    if args.platform == 'ascend9382' and not args.figures_only:
        for source in (args.ascend_root, args.npu_smi, args.ascend_install_info):
            if not source.exists():
                raise SystemExit('Missing Ascend host mount: ' + str(source))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # A list, never shell=True: spaces or shell characters in output paths stay literal.
    try:
        return subprocess.run(command, check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
