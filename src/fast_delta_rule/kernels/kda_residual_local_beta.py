"""Local P reuse and complete beta gradient; paired with kda_intra_beta."""
import torch
import triton
from .kda_residual_tri_beta import residual_tri_beta_producer
from .kda_residual_local_beta_consumer import residual_wy_local_beta_consumer


class LocalBetaResidual:
    best_config='triangular P BR32; K-fast BK32 BV32; total beta in producer; 2 kernels'
    def __getitem__(self,grid):
        def launch(**kw):
            p=torch.empty_like(kw['dv'])
            nk=triton.cdiv(kw['K'],32)
            self.producer=residual_tri_beta_producer[(grid[0]*(kw['BT']//32),grid[1])](
                kw['v'],kw['v_new'],kw['A'],kw['dv'],kw['beta'],p,kw['dv2'],kw['db'],kw['dA'],
                kw['cu_seqlens'],kw['chunk_indices'],kw['T'],HV=kw['HV'],V=kw['V'],
                VP=triton.next_power_of_2(kw['V']),BT=kw['BT'],BR=32,
                IS_VARLEN=kw['cu_seqlens'] is not None,USE_GRAPH=kw['USE_GRAPH'],ROW_FAST=True,
                num_warps=4,num_stages=2)
            self.consumer=residual_wy_local_beta_consumer[(grid[0]*nk,grid[1])](
                **dict(kw,dv=p,db_v=kw['db']),TOTAL_TOKENS=kw['v'].shape[0]*kw['T'],
                BK=32,BV=32,num_warps=4,num_stages=2)
        return launch
