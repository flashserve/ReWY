"""Device launch helpers, independent of the experiment harness."""
import torch
import triton

DEV = 'npu' if hasattr(torch, 'npu') and torch.npu.is_available() else 'cuda'

def launch(kernel, tasks, **kwargs):
    if DEV == 'npu':
        from fla.utils import ascend_compile_kwargs
        from fla.utils.ascend_ub_manager import get_npu_properties
        cores = get_npu_properties()['num_aicore']
        kernel[(cores,)](TASKS=tasks, STEP=cores, **kwargs, **ascend_compile_kwargs())
    else:
        kernel[(tasks,)](TASKS=tasks, STEP=tasks, **kwargs, num_warps=4, num_stages=2)
