"""CPU-only checks for corrected routing, precision guards and source seals."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fast_delta_rule.backends import npu_target
from fast_delta_rule.kernels import matrix_intra_npu950 as entry

ROOT=Path(__file__).resolve().parents[1]


def script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def inputs(monkeypatch,family):
    mock_torch=SimpleNamespace(float32='f32',bfloat16='bf16')
    monkeypatch.setitem(sys.modules,'torch',mock_torch)
    device='npu:0';shape=(1,64,4,128)
    def tensor(shape,dtype='f32'):
        return SimpleNamespace(shape=shape,dtype=dtype,device=device,is_contiguous=lambda:True)
    x={n:tensor(shape) for n in ('g','dq','dk','dg')}
    x.update(q=tensor(shape,'bf16'),k=tensor(shape,'bf16'),
             dAqk=tensor((1,64,4,64)),dAkk=tensor((1,64,4,64)))
    beta_shape=shape[:-1] if family=='kda' else shape
    x['beta' if family=='kda' else 'b']=tensor(beta_shape,'f32' if family=='kda' else 'bf16')
    x['db']=tensor(beta_shape)
    return x


@pytest.mark.parametrize('family',['kda','gdn2'])
def test_new_950_entry_reaches_corrected_dataflow(monkeypatch,family):
    x=inputs(monkeypatch,family);guard=[];calls=[]
    monkeypatch.setattr(npu_target,'require_950',lambda device:guard.append(device))
    def corrected(*args,**kw):
        calls.append((args,kw));return 'corrected-output'
    monkeypatch.setitem(sys.modules,f'fast_delta_rule.kernels.{family}_intra_npu9382',
                        SimpleNamespace(**{family+'_intra':corrected}))
    assert getattr(entry,family+'_intra')(**x,use_graph=False)=='corrected-output'
    assert guard==['npu:0'] and len(calls)==1
    assert calls[0][0][3] is x['beta' if family=='kda' else 'b']
    assert not hasattr(entry,'matrix_kernel'), 'Old C64/K32 kernel must not remain selected'


@pytest.mark.parametrize('family',['kda','gdn2'])
def test_950_rejects_lower_precision_original_coefficients(monkeypatch,family):
    x=inputs(monkeypatch,family)
    monkeypatch.setattr(npu_target,'require_950',lambda device:None)
    x['dAqk'].dtype='bf16'
    with pytest.raises(ValueError,match='FP32'):
        getattr(entry,family+'_intra')(**x)


def test_source_seal_rejects_old_or_modified_code(tmp_path,monkeypatch):
    runner=script('npu950_experiment');current={'src/kernel.py':'new-sha'}
    monkeypatch.setattr(runner,'source_hashes',lambda:current)
    path=tmp_path/'manifest.json'
    path.write_text(json.dumps({'revision':runner.PREPARED_REVISION,'source_hashes':current}))
    assert runner.check_prepared(path)['verified']
    current['src/kernel.py']='modified-sha'
    with pytest.raises(RuntimeError,match='source mismatch'):runner.check_prepared(path)


@pytest.mark.parametrize('family',['kda','gdn2'])
def test_real_fixture_oracle_is_finite_for_wide_gates(family):
    regression=script('npu950_intra_regression')
    data,metadata=regression.load_fixture(family)
    assert data['k'].shape==(1,64,4,128)
    assert metadata['dtypes']['dAqk']=='torch.float32'
    data['g']*=10.
    result=regression.oracle(data,family)
    assert all(np.isfinite(x).all() for x in result)
