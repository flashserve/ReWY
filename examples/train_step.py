"""Unmodified FLA training call; choose its backend with `fdr run`."""
import argparse
import torch
from fla.ops.kda import chunk_kda

p=argparse.ArgumentParser()
p.add_argument('--steps',type=int,default=3)
p.add_argument('--device',default='cuda')
a=p.parse_args()
torch.manual_seed(0)
q=torch.nn.Parameter(torch.randn(1,128,2,128,device=a.device,dtype=torch.bfloat16))
k=torch.nn.functional.normalize(torch.randn_like(q),dim=-1)
v=torch.randn_like(q)
g=-torch.rand(q.shape,device=a.device)*.05
beta=torch.rand(q.shape[:-1],device=a.device,dtype=q.dtype)
optimizer=torch.optim.SGD([q],lr=1e-3)
for step in range(a.steps):
    optimizer.zero_grad(set_to_none=True)
    out,_=chunk_kda(q,k,v,g,beta)
    loss=out.float().square().mean()
    loss.backward()
    optimizer.step()
    print(f'step={step} loss={loss.item():.6f}')
