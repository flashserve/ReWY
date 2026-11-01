"""Numerical drift must fail even when platform-dependent raster bytes are allowed."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('artifact_check', ROOT / 'scripts/check_reproduction.py')
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.fixture
def rendered(tmp_path, monkeypatch):
    paper, output = tmp_path / 'paper', tmp_path / 'output'
    paper.mkdir()
    (paper / 'data').mkdir()
    output.mkdir()
    for name in ('figure_data.json', 'intro_graph_data.json'):
        for folder in (paper / 'data', output):
            (folder / name).write_text(json.dumps({'speedup': 1.25}))
    (output / 'figure.png').write_bytes(b'different platform raster bytes')
    monkeypatch.setattr(checker, 'BENCHMARKS', paper)
    return output, {'figure_png_sha256': {'figure.png': checker.digest(b'original raster bytes')}}


def test_equal_data_can_report_different_raster_bytes(rendered):
    assert checker.verify_rendered(*rendered) == ['figure.png']


def test_strict_mode_still_requires_original_bytes(rendered):
    with pytest.raises(AssertionError, match='bytes differ'):
        checker.verify_rendered(*rendered, strict_bytes=True)


def test_changed_numerical_values_always_fail(rendered):
    output, manifest = rendered
    (output / 'figure_data.json').write_text(json.dumps({'speedup': 9.99}))
    with pytest.raises(AssertionError, match='values differ'):
        checker.verify_rendered(output, manifest)
