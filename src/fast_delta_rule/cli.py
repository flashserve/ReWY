"""Command-line bootstrap and environment diagnostics."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import sys

def doctor(check_sources=False):
    result={'python':sys.version.split()[0], 'platform':sys.platform}
    for name in ('fast-delta-rule','torch','triton','triton-ascend','torch-npu','flash-linear-attention','transformers'):
        try: result[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result[name]=None
    if check_sources:
        from .runtime import check_baseline
        result['fla_commit']=check_baseline()
        import torch
        result['cuda_available']=torch.cuda.is_available()
        if torch.cuda.is_available(): result['device']=torch.cuda.get_device_name()
        if hasattr(torch,'npu') and torch.npu.is_available(): result['device']=torch.npu.get_device_name()
    return result

def main(argv=None):
    parser=argparse.ArgumentParser(prog='fdr',description='Select delta-rule backends without editing your training script.')
    sub=parser.add_subparsers(dest='command',required=True)
    d=sub.add_parser('doctor');d.add_argument('--check-sources',action='store_true')
    r=sub.add_parser('run')
    r.add_argument('--backend',choices=['original','optimized'],default=os.getenv('FDR_BACKEND','optimized'))
    r.add_argument('--fallback',choices=['error','original'],default='error')
    r.add_argument('--ops',default='kda,gdn,gdn2,delta')
    r.add_argument('--integration',action='append',choices=['fla','qwen3_5','gdn2'],default=['fla'])
    r.add_argument('--report',type=Path)
    r.add_argument('--allow-no-backward',action='store_true')
    r.add_argument('script');r.add_argument('args',nargs=argparse.REMAINDER)
    v=sub.add_parser('verify')
    v.add_argument('--device',choices=['cuda','npu'],default='cuda')
    v.add_argument('--ops',default='kda,gdn,gdn2,delta')
    v.add_argument('--output',type=Path,default=Path('results/verification.json'))
    b=sub.add_parser('benchmark')
    b.add_argument('--device',choices=['cuda','npu'],default='cuda')
    b.add_argument('--op',choices=['kda','gdn','gdn2','delta'],default='kda')
    b.add_argument('--length',type=int,default=2048)
    b.add_argument('--heads',type=int,default=16)
    b.add_argument('--batch',type=int,default=1)
    b.add_argument('--rounds',type=int,default=30)
    b.add_argument('--boundary',choices=['B','FB'],default='B')
    b.add_argument('--policy',choices=['recompute','all'],default='recompute')
    b.add_argument('--output',type=Path,default=Path('results/benchmark.json'))
    suite=sub.add_parser('reproduce')
    suite.add_argument('--suite',choices=['release'],default='release')
    suite.add_argument('--device',choices=['cuda','npu'],default='cuda')
    suite.add_argument('--output',type=Path,default=Path('results/release'))
    a=parser.parse_args(argv)
    if a.command=='doctor':
        print(json.dumps(doctor(a.check_sources),indent=2));return
    if a.command=='verify':
        from .validation import verify
        result=verify(a.device,a.ops.split(','))
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
        if not result['passed']: raise SystemExit(1)
        return
    if a.command=='benchmark':
        from .benchmark import benchmark
        result=benchmark(a.device,a.op,a.length,a.heads,a.rounds,a.boundary,a.policy,a.batch)
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2));return
    if a.command=='reproduce':
        from .reproduce import reproduce
        reproduce(a.device,a.output);return
    from . import enable,stats
    ops=tuple(a.ops.split(','))
    enable(a.backend,strict=a.fallback=='error',ops=ops,integrations=a.integration)
    stats(reset=True)
    print(f'[fast-delta-rule] backend={a.backend} ops={",".join(ops)} fallback={a.fallback}',file=sys.stderr)
    old_argv=sys.argv[:];old_path=sys.path[:]
    sys.argv=[a.script]+a.args
    sys.path.insert(0,str(Path(a.script).resolve().parent))
    try:
        try: runpy.run_path(a.script,run_name='__main__')
        except SystemExit as e:
            if e.code not in (None,0): raise
        counts=stats()
        if a.backend=='optimized' and not a.allow_no_backward and not any(
            counts.get(op+'.optimized_backward',0) for op in ops):
            raise RuntimeError('No optimized backward ran. Check --integration/--ops; use --allow-no-backward only for forward-only scripts.')
    finally:
        result={'backend':a.backend,'counts':stats(),'count_boundary':'Python submissions; graph replays are not additional submissions'}
        print('[fast-delta-rule] '+json.dumps(result),file=sys.stderr)
        if a.report:
            a.report.parent.mkdir(parents=True,exist_ok=True)
            a.report.write_text(json.dumps(result,indent=2)+'\n')
        sys.argv=old_argv;sys.path[:]=old_path

if __name__=='__main__': main()
