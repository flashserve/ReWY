import subprocess
import sys
import pytest
from fast_delta_rule import runtime

def test_import_has_no_framework_side_effects():
    subprocess.run([sys.executable,'-c',"import fast_delta_rule,sys; assert 'torch' not in sys.modules; assert 'fla' not in sys.modules"],check=True)

def test_nested_selection_restores_on_exception():
    assert runtime._selection.get().backend=='original'
    with runtime.use_backend('optimized',ops=('kda',)):
        assert runtime._selection.get().ops==('kda',)
        with pytest.raises(RuntimeError):
            with runtime.use_backend('original'):
                raise RuntimeError('abort')
        assert runtime._selection.get().backend=='optimized'
    assert runtime._selection.get().backend=='original'

def test_unknown_backend_and_operator_fail():
    with pytest.raises(ValueError):runtime.configure('typo')
    with pytest.raises(ValueError):runtime.configure(ops=('unknown',))

def test_no_hit_is_an_error():
    runtime.stats(reset=True)
    with pytest.raises(RuntimeError,match='No optimized_backward'):runtime.assert_used(('kda',))
    runtime.record('kda','optimized_forward')
    with pytest.raises(RuntimeError):runtime.assert_used(('kda',))
    runtime.record('kda','optimized_backward')
    runtime.assert_used(('kda',))
    runtime.stats(reset=True)

def test_cli_help_without_accelerator():
    subprocess.run([sys.executable,'-m','fast_delta_rule','--help'],check=True,capture_output=True)
