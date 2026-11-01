"""Paper-only shared inputs, independent FP64 recurrence and error metrics."""
import importlib, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
DEVICE=os.environ.get('KDA_DEVICE','cuda')
if DEVICE=='cuda': os.environ['FLA_DISABLE_BACKEND_DISPATCH']='1'
sys.path.insert(0,str(ROOT/'vendor/fla'))
import torch
import torch.nn.functional as F
import triton
if DEVICE=='npu':
    import torch_npu
    torch.npu.set_device(0)
    npu_bwd=importlib.import_module('fla.ops.kda.backends.triton_ascend.chunk_bwd')
bwd=importlib.import_module('fla.ops.kda.chunk_bwd')
ACCEL=torch.cuda if DEVICE=='cuda' else torch.npu
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False


def inputs(B, T, H, D, HV=None, V=None, seed=42, gate_div=16, dtype=torch.bfloat16, state_v_first=False):
    HV, V = HV or H, V or D
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, D, device=DEVICE), dim=-1).to(dtype)
    k = F.normalize(torch.randn_like(q, dtype=torch.float32), dim=-1).to(dtype)
    v = torch.randn(B, T, HV, V, dtype=dtype, device=DEVICE)
    g = F.logsigmoid(torch.randn(B, T, HV, D, device=DEVICE)) / gate_div
    beta = torch.rand(B, T, HV, dtype=dtype, device=DEVICE) * 0.8 + 0.1
    h0 = torch.randn(B, HV, D, V, device=DEVICE) * 0.1
    if state_v_first:
        h0 = h0.transpose(-1, -2).contiguous()
    xs = tuple(x.requires_grad_() for x in (q,k,v,g,beta,h0))
    do = torch.randn_like(v)
    dht = torch.randn_like(h0) * 0.1
    return xs, (do,dht)

def oracle(xs, state_v_first=False):
    """Independent token recurrence in CPU FP64, including final-state gradient."""
    q,k,v,g,beta,h = xs
    if state_v_first:
        h = h.transpose(-1,-2)
    G = v.shape[2] // q.shape[2]
    q = q.repeat_interleave(G,dim=2) * q.shape[-1]**-0.5
    k = k.repeat_interleave(G,dim=2)
    out = []
    for t in range(q.shape[1]):
        h = h * g[:,t].exp().unsqueeze(-1)
        residual = v[:,t] - (k[:,t].unsqueeze(-1)*h).sum(-2)
        h = h + (beta[:,t].unsqueeze(-1) * k[:,t]).unsqueeze(-1) * residual.unsqueeze(-2)
        out.append((q[:,t].unsqueeze(-1)*h).sum(-2))
    return torch.stack(out, dim=1), h.transpose(-1,-2) if state_v_first else h

def error(ref, actual, budget):
    ref, actual = ref.detach().cpu().double(), actual.detach().cpu().double()
    diff = actual-ref
    rel = float(diff.norm() / ref.norm().clamp_min(1e-12))
    maxabs = float(diff.abs().max())
    return {'rel_l2':rel, 'max_abs':maxabs, 'mean_abs':float(diff.abs().mean()),
            'finite':bool(actual.isfinite().all()), 'budget_rel_l2':budget,
            'pass':bool(actual.isfinite().all() and (rel < budget or maxabs < 1e-6))}

def emit(record, path):
    record['time'] = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    with open(path,'a') as f:
        f.write(json.dumps(record)+'\n')
    print(json.dumps(record), flush=True)
