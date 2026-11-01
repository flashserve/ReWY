"""Real captured FLA intra inputs, independent FP64 oracle, no timing.

Only run() imports torch and executes NPU code. Fixtures are losslessly stored
as float32 NumPy arrays; original BF16 tensors are restored before execution.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / 'validation/npu950_intra'


def load_fixture(family):
    metadata = json.loads((FIXTURES / 'metadata.json').read_text(encoding='utf-8'))[family]
    path = FIXTURES / f'{family}.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest() != metadata['fixture_sha256']:
        raise RuntimeError(f'{family} real-input fixture hash mismatch')
    with np.load(path, allow_pickle=False) as data:
        arrays = {k: data[k].copy() for k in data.files}
    return arrays, metadata


def oracle(x, family):
    """Explicit pairwise gate contraction, independent of tiled NPU dataflow."""
    q,k,g=(x[n].astype(np.float64) for n in ('q','k','g'))
    beta=x['beta' if family=='kda' else 'b'].astype(np.float64)
    outputs=[x[n].astype(np.float64).copy() for n in ('dq','dk','db','dg')]
    B,T,H,K=q.shape
    lower=np.tri(T,dtype=bool)
    for b in range(B):
        for h in range(H):
            gh=g[b,:,h]
            gate=np.exp2(np.where(lower[:,:,None],gh[:,None,:]-gh[None,:,:],0.))
            l=np.tril(x['dAqk'][b,:,h].astype(np.float64))
            d=np.tril(x['dAkk'][b,:,h].astype(np.float64),-1)
            be=beta[b,:,h,None] if family=='kda' else beta[b,:,h]
            uq=np.einsum('ij,ijk,jk->ik',l,gate,k[b,:,h])
            ue=np.einsum('ij,ijk,jk->ik',d,gate,k[b,:,h])
            uk=np.einsum('ji,jik,jk->ik',l,gate,q[b,:,h])+np.einsum('ji,jik,jk->ik',d,gate,k[b,:,h]*be)
            outputs[0][b,:,h]+=uq
            outputs[1][b,:,h]+=be*ue+uk
            outputs[2][b,:,h]+=(ue*k[b,:,h]).sum(-1) if family=='kda' else ue*k[b,:,h]
            outputs[3][b,:,h]+=q[b,:,h]*uq+k[b,:,h]*(be*ue-uk)
    return outputs


def run(ops, output):
    import torch
    from fast_delta_rule.backends.npu_target import require_950
    from fast_delta_rule.kernels import matrix_intra_npu950 as impl
    from fast_delta_rule.validation import error
    require_950()
    result={'revision':impl.SCHEDULE_REVISION,'scope':'matrix intra only; not complete B/FB or performance',
            'passed':True,'cases':[]}
    for family in (op for op in ops if op in ('kda','gdn2')):
        data,metadata=load_fixture(family)
        for scale,batch in ((1.,1),(.2,1),(3.,1),(10.,1),(1.,2)):
            cpu={k:np.repeat(v,batch,axis=0) for k,v in data.items()}
            cpu['g']*=scale
            reference=oracle(cpu,family)
            kw={k:torch.from_numpy(v).to(dtype=getattr(torch,metadata['dtypes'][k].split('.')[-1]),device='npu') for k,v in cpu.items()}
            actual=getattr(impl,family+'_intra')(**kw)
            torch.npu.synchronize()
            expected_dtypes=(torch.float32,torch.float32,torch.float32,torch.float32) if family=='kda' else (torch.bfloat16,torch.bfloat16,torch.float32,torch.float32)
            metrics={}
            for name,ref,got,dtype in zip(('dq','dk','db','dg'),reference,actual,expected_dtypes):
                if got.dtype!=dtype or tuple(got.shape)!=ref.shape:
                    raise RuntimeError(f'{family} {name}: output dtype/shape changed')
                rounded=torch.from_numpy(ref).to(dtype=dtype)
                metrics[name]=error(rounded,got,.02 if name in ('db','dg') else .008)
            row={'family':family,'gate_scale':scale,'batch':batch,'metrics':metrics,
                 'passed':all(v['passed'] for v in metrics.values())}
            result['cases'].append(row);result['passed'] &= row['passed']
            output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result
