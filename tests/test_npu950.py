"""CPU-only routing and independent mathematical checks, not NPU acceptance."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from fast_delta_rule.backends import backward, npu_target


@pytest.mark.parametrize('name', ['Ascend950PR', 'Ascend950DT', 'Ascend950', 'Ascend 950PR'])
def test_950_physical_names(name):
    assert npu_target.classify_device(name) == 'npu950'


def test_legacy_device_is_not_relabelled():
    assert npu_target.classify_device('Ascend910_9382') == 'npu'
    for name in ('950', 'NVIDIA A800', 'Ascend910B', 'unknown', 'Ascend950/mocked'):
        with pytest.raises(RuntimeError):
            npu_target.classify_device(name)


@pytest.mark.parametrize('op', ['kda', 'gdn2'])
@pytest.mark.parametrize('variant', ['residual', 'matrix', 'combined'])
def test_950_vector_bindings_never_import_native_extension(monkeypatch, op, variant):
    wy_name = 'chunk_kda_bwd_wy_dqkg_fused' if op == 'kda' else 'chunk_gdn2_bwd_wy_dqkg_fused'
    intra_name = f'chunk_{op}_bwd_intra'
    original_intra, port_intra, original_wy, port_wy = [object() for _ in range(4)]
    base = SimpleNamespace(**{intra_name: original_intra, wy_name: original_wy,
                              f'chunk_{op}_bwd': object()})
    monkeypatch.setattr(backward.importlib, 'import_module', lambda name: base)
    monkeypatch.setattr(backward, '_clone', lambda original, overrides: SimpleNamespace(overrides=overrides))
    monkeypatch.setitem(sys.modules, 'fast_delta_rule.kernels.kda_residual_npu_onchip',
                        SimpleNamespace(chunk_kda_bwd_wy_dqkg_fused_npu=port_wy))
    monkeypatch.setitem(sys.modules, 'fast_delta_rule.kernels.gdn2_full_npu', SimpleNamespace(residual=port_wy))
    monkeypatch.setitem(sys.modules, 'fast_delta_rule.kernels.matrix_intra_npu950',
                        SimpleNamespace(kda_intra=port_intra, gdn2_intra=port_intra))
    monkeypatch.setitem(sys.modules, 'fast_delta_rule.backends.npu_native', None)
    backward._vector.cache_clear()
    result = backward._vector(op, 'npu950', variant)
    assert (wy_name in result.overrides) == (variant != 'matrix')
    assert (intra_name in result.overrides) == (variant != 'residual')
    if wy_name in result.overrides:
        assert result.overrides[wy_name] is port_wy
    if intra_name in result.overrides:
        assert result.overrides[intra_name] is port_intra
    backward._vector.cache_clear()


def _objective(q, k, gate, beta, l, d):
    # Independent scalar forward, gates in natural-log units. D is strict
    # lower triangular; L includes the diagonal. No residual implementation.
    value = 0.
    for i in range(len(q)):
        for j in range(i + 1):
            fac = np.exp(gate[i] - gate[j])
            value += l[i, j] * np.sum(q[i] * k[j] * fac)
            if j < i:
                value += d[i, j] * np.sum(beta[i] * k[i] * k[j] * fac)
    return value


@pytest.mark.parametrize('vector_beta', [False, True])
@pytest.mark.parametrize('span', [3., 96.])
def test_matrix_intra_vjp_against_scalar_forward_finite_difference(vector_beta, span):
    rng = np.random.default_rng(950)
    c, kdim = 16, 32
    q, k = rng.normal(size=(2, c, kdim))
    gate = -np.linspace(0., span * np.log(2), c)[:, None] * np.ones((1, kdim))
    beta = rng.uniform(.1, .9, (c, kdim if vector_beta else 1))
    beta[::3] = 0.  # No divide-by-beta shortcut.
    l = np.tril(rng.normal(size=(c, c)))
    d = np.tril(rng.normal(size=(c, c)), -1)
    if span < 80:
        center = (gate.min(0) + gate.max(0)) / 2
        plus, minus = np.exp(gate - center), np.exp(center - gate)
        uq, ue = (l @ (k * minus)) * plus, (d @ (k * minus)) * plus
        uk = (l.T @ (q * plus) + d.T @ (k * beta * plus)) * minus
    else:
        # Direct pairwise contraction used when centered factors are unsafe.
        fac = np.exp(gate[:, None, :] - gate[None, :, :])
        uq = np.einsum('ij,jk,ijk->ik', l, k, fac)
        ue = np.einsum('ij,jk,ijk->ik', d, k, fac)
        uk = np.einsum('ji,jk,jik->ik', l, q, fac) + np.einsum('ji,jk,jik->ik', d, beta * k, fac)
    db = ue * k if vector_beta else (ue * k).sum(1, keepdims=True)
    gradients = (uq, beta * ue + uk, q * uq + k * (beta * ue - uk), db)
    inputs = (q, k, gate, beta)
    for i, grad in enumerate(gradients):
        direction = rng.normal(size=inputs[i].shape)
        positive, negative = list(inputs), list(inputs)
        eps = 1e-5
        positive[i] = inputs[i] + eps * direction
        negative[i] = inputs[i] - eps * direction
        numerical = (_objective(*positive, l, d) - _objective(*negative, l, d)) / (2 * eps)
        analytical = np.sum(grad * direction)
        np.testing.assert_allclose(numerical, analytical, rtol=2e-6, atol=2e-6)


def test_experiment_plan_keeps_policy_and_boundary_distinctions():
    path = Path(__file__).resolve().parents[1] / 'scripts/npu950_experiment.py'
    spec = importlib.util.spec_from_file_location('experiment950', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    plan = module.plan(['kda', 'gdn'], 'combined')
    assert plan['policies'] == {'kda': ['recompute', 'all'], 'gdn': ['recompute']}
    assert plan['boundaries'] == ['B', 'FB']
    assert len(plan['shapes_B_T_H']) == 6
    with pytest.raises(ValueError):
        module.plan(['delta'], 'matrix')
