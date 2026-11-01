#!/usr/bin/env python3
"""Create a Docker build context from explicit measured-runtime exports; CPU only.

Runtime exports are deliberately external, never downloaded from an unpinned pip
resolution. See README.md for export layout and recorded software requirements.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', required=True, choices=['a800', 'ascend9382'])
    parser.add_argument('--context', type=Path, required=True)
    parser.add_argument('--fla', type=Path, required=True, help='Frozen FLA checkout root, containing fla/ and LICENSE')
    args = parser.parse_args()
    dest = args.context.resolve()
    baseline = json.loads((ROOT / 'src/fast_delta_rule/baseline.json').read_text())
    for name, expected in baseline['files'].items():
        if hashlib.sha256((args.fla / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != expected:
            raise ValueError('FLA source mismatch: ' + name)
    inputs = ['runtime.tar.gz']
    if args.platform == 'ascend9382':
        inputs += ['model-deps.tar.gz', 'plot-deps.tar.gz', 'toolchain.tar.gz', 'libkda_residual_native.so']
    for name in inputs:
        if not (dest / name).is_file():
            raise FileNotFoundError(dest / name)
    names = {'LICENSE', 'THIRD_PARTY_NOTICES.md', 'pyproject.toml'}
    for folder in ('src', 'research', 'benchmarks', 'scripts', 'containers', 'validation', 'docs/figures'):
        names |= {
            p.relative_to(ROOT).as_posix() for p in (ROOT / folder).rglob('*')
            if p.is_file() and not {'__pycache__', 'results', 'build'} & set(p.parts)
            and p.suffix not in ('.pyc', '.log')
        }
    with tarfile.open(dest / 'artifact.tar.gz', 'w:gz') as tar:
        for name in sorted(names):
            tar.add(ROOT / name, arcname=name, recursive=False)
    with tarfile.open(dest / 'fla.tar.gz', 'w:gz') as tar:
        for path in sorted((args.fla / 'fla').rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                tar.add(path, arcname=path.relative_to(args.fla).as_posix())
        tar.add(args.fla / 'LICENSE', arcname='LICENSE')
    for original, name in [('Dockerfile.' + args.platform, 'Dockerfile'), ('entrypoint.sh', 'entrypoint.sh')]:
        (dest / name).write_bytes((ROOT / 'containers' / original).read_bytes().replace(b'\r\n', b'\n'))
    record = dict(platform=args.platform, frozen_fla_commit=baseline['commit'],
                  source_files=len(names), inputs={n: sha(dest / n) for n in inputs + ['artifact.tar.gz', 'fla.tar.gz']})
    (dest / 'runtime-manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
