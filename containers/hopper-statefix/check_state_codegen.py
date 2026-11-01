"""Compile real state launch specializations for SM90 without a CUDA device.

JITFunction.warmup constructs Triton's actual pointer-alignment attributes.
Hand-written ASTSource attributes keyed by argument names do not exercise the
same compilation path and previously hid the D256 shared-memory overflow.
"""
import argparse
from contextlib import ExitStack
import importlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
parser.add_argument('--report',type=Path)
args=parser.parse_args()
sys.path.insert(0,str(args.root))
import check_hopper_compat as compat

def compile_states():
    import torch
    import triton
    from operators import Operator
    from kernels import scalar_state_reference_gpu as scalar
    from kernels import vector_state_gpu as vector
    triton.runtime.driver.active.get_current_stream=lambda device:0
    rows=[]
    limit=232448
    # The fake driver cannot load a module or launch any kernel. warmup only
    # produces PTX/cubin; assertions below never initialize compiled handles.
    class CompileLaunch:
        def __init__(self,fn,label,case,control=False):
            self.fn,self.label,self.case,self.control=fn,label,case,control
        def __getitem__(self,grid):
            def launch(*pos,**kw):
                if self.control:kw.pop('arch',None)
                kernel=self.fn.warmup(*pos,**kw,grid=grid)
                shared=kernel.metadata.shared
                row=dict(family=self.label,case=self.case,shared_bytes=shared,limit=limit,
                         stages=kw['num_stages'],warps=kw['num_warps'],arch=kw.get('arch','sm90'),control=self.control,
                         gpu_executed=False)
                assert kernel.asm['cubin'] and '.target sm_90' in kernel.asm['ptx']
                rows.append(row)
                if self.control:
                    assert shared>limit,('Historical reproducer stopped failing',row)
                else:
                    assert shared<=limit,('State kernel shared-memory overflow',row)
                    if self.label.startswith('delta'):
                        assert kw['arch']=='sm80' and 'mma.sync.' in kernel.asm['ptx'] and 'wgmma.' not in kernel.asm['ptx']
                        from triton.backends.compiler import GPUTarget
                        options=dict(num_warps=kw['num_warps'],num_stages=kw['num_stages'])
                        native=triton.compile(kernel.src,target=GPUTarget('cuda',80,32),options=options)
                        explicit=triton.compile(kernel.src,target=GPUTarget('cuda',80,32),options=dict(options,arch='sm80'))
                        assert native.asm['cubin']==explicit.asm['cubin'],'A800 machine code changed'
                        row['a800_machine_code_unchanged']=True
                        row['a800_shared_bytes']=native.metadata.shared
            return launch
    def tensors(dim,dtype,svf=False,states=True,varlen=False):
        value=80 if dim==96 else dim
        q=torch.empty((1,129,2,dim),device='meta',dtype=dtype)
        v=torch.empty((1,129,2,value),device='meta',dtype=dtype)
        h=torch.empty((1,2,*((value,dim) if svf else (dim,value))),device='meta',dtype=torch.float32)
        kw=dict(q=q,k=q,w=q,do=v,dv=v,h0=h if states else None,dht=h if states else None,
                scale=dim**-.5,state_v_first=svf,value_block=16)
        if varlen:
            kw.update(cu_seqlens=torch.empty(3,device='meta',dtype=torch.int64),
                      chunk_indices=torch.empty((3,2),device='meta',dtype=torch.int32),
                      chunk_offsets=torch.empty(3,device='meta',dtype=torch.int64))
        return kw
    # Exercise the actual Operator.backward caller, including its explicit
    # schedule arguments. This fails against the unmodified syncmma1 image.
    kw=tensors(256,torch.bfloat16)
    op=Operator('delta');q,v,h=kw['q'],kw['do'],kw['h0']
    consumer=importlib.import_module('kernels.scalar_consumer_schedule_gpu')
    with patch.object(scalar,'reference_kernel',CompileLaunch(scalar.reference_kernel,'delta_actual_backward',dict(D=256))), \
         patch.object(op.c,'chunk_bwd_dv_local',return_value=v), \
         patch.object(consumer,'adjoint',return_value=(None,)*5):
        op.backward((q,q,v,q,h),(v,h),((None,q),dict(wu=(q,v),hr=(q,v,None)),()),'optimized','all')
    # Retain an exact negative control so a disabled pipeline cannot make the
    # test falsely green again.
    with patch.object(scalar,'reference_kernel',CompileLaunch(scalar.reference_kernel,'delta_old_s3',dict(D=256),True)):
        scalar.state_bwd(**kw,stages=3)
    for dim in (64,96,128,192,256):
        for dtype in (torch.bfloat16,torch.float16):
            for svf in (False,True):
                kw=tensors(dim,dtype,svf)
                case=dict(D=dim,dtype=str(dtype),svf=svf,states=True,varlen=False)
                with patch.object(scalar,'reference_kernel',CompileLaunch(scalar.reference_kernel,'delta',case)):
                    scalar.state_bwd(**kw)
    for dtype in (torch.bfloat16,torch.float16):
        for svf in (False,True):
            for states in (False,True):
                for varlen in (False,True):
                    kw=tensors(256,dtype,svf,states,varlen)
                    case=dict(D=256,dtype=str(dtype),svf=svf,states=states,varlen=varlen)
                    with patch.object(scalar,'reference_kernel',CompileLaunch(scalar.reference_kernel,'delta_optional',case)):
                        scalar.state_bwd(**kw)
    # The other optimized state implementation is unchanged, but also cover
    # D256 through its real launch rather than a synthetic signature.
    for dim in (128,256):
        for dtype in (torch.bfloat16,torch.float16):
            kw=tensors(dim,dtype)
            kw['gk']=torch.empty_like(kw['q'],dtype=torch.float32)
            with patch.object(vector,'state_kernel',CompileLaunch(vector.state_kernel,'gdn2',dict(D=dim,dtype=str(dtype)))):
                vector.state_bwd(**kw)
    result=dict(passed=True,compiled=len(rows),positive_checks=len(rows)-1,
                max_shared_bytes=max(r['shared_bytes'] for r in rows if not r['control']),
                historical_overflow_reproduced=True,gpu_executed=False,rows=rows)
    if args.report:args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    return rows

compat.compile_sm90=compile_states
sys.argv=[sys.argv[0],'--root',str(args.root),'--vendor','single','--gpu','H100','--compile-sm90']
compat.main()
