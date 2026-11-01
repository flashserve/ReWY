"""CPU-only dispatch guard: 9382 must not select the historical BF16-L/D paths."""
import sys
from types import SimpleNamespace
import pytest
from fast_delta_rule.backends import backward


@pytest.mark.parametrize('op', ['kda', 'gdn2'])
def test_9382_uses_corrected_fp32_matrix(monkeypatch, op):
    wy_name=f'chunk_{op}_bwd_wy_dqkg_fused'
    intra_name=f'chunk_{op}_bwd_intra'
    original_intra, original_wy, fixed_intra, residual = [object() for _ in range(4)]
    base=SimpleNamespace(**{intra_name:original_intra,wy_name:original_wy,f'chunk_{op}_bwd':object()})
    monkeypatch.setattr(backward.importlib,'import_module',lambda name:base)
    monkeypatch.setattr(backward,'_clone',lambda original,overrides:SimpleNamespace(overrides=overrides))
    monkeypatch.setitem(sys.modules,'fast_delta_rule.kernels.kda_residual_npu_onchip',
                        SimpleNamespace(chunk_kda_bwd_wy_dqkg_fused_npu=residual))
    monkeypatch.setitem(sys.modules,'fast_delta_rule.kernels.gdn2_full_npu',SimpleNamespace(residual=residual))
    monkeypatch.setitem(sys.modules,f'fast_delta_rule.kernels.{op}_intra_npu9382',
                        SimpleNamespace(**{op+'_intra':fixed_intra}))
    monkeypatch.setitem(sys.modules,'fast_delta_rule.backends.npu_native',None)
    monkeypatch.setitem(sys.modules,'fast_delta_rule.kernels.gdn2_kernels',None)
    backward._vector.cache_clear()
    result=backward._vector(op,'npu')
    assert result.overrides[intra_name] is fixed_intra
    assert result.overrides[wy_name] is residual
    backward._vector.cache_clear()
