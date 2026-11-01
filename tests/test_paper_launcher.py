"""CPU-only contracts for the host launcher; never starts Docker or a device."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('paper_launcher', ROOT / 'scripts/reproduce_paper.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_image_pins_are_immutable():
    images = json.loads((ROOT / 'containers/images.json').read_text())
    expected = {
        'a800': 'sha256:6ff3251ceea9f227b48aa09cf7f44e58efc7ad4824c2aabea28e4bce1c694631',
        'ascend9382': 'sha256:d4374e36f6cff239e3d836d75c3e7bc01f55cbd086d0ef23f41d79f947df80c5',
    }
    assert {platform: entry['digest'] for platform, entry in images.items()} == expected


@pytest.mark.parametrize('platform', ['a800', 'ascend9382'])
def test_figure_plan_never_mounts_devices_or_creates_output(platform, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(launcher.subprocess, 'run', lambda *a, **k: pytest.fail('Plan invoked Docker'))
    output = tmp_path / 'missing parent' / 'figures'
    assert launcher.main([platform, '--figures-only', '--output', str(output), '--plan']) == 0
    assert not output.parent.exists()
    text = capsys.readouterr().out
    assert 'replay' in text and '@sha256:' in text
    assert '--gpus' not in text and '--privileged' not in text and 'ASCEND_RT_VISIBLE_DEVICES' not in text


def test_a800_remaps_one_physical_device_and_preserves_literal_paths(tmp_path):
    output = tmp_path / 'space ; shell $ chars' / 'run 01'
    args = launcher.parse_args(['a800', '--device', '3', '--output', str(output), '--resume'])
    command = launcher.docker_command(args)
    assert command[command.index('--gpus') + 1] == 'device=3'
    assert 'CUDA_VISIBLE_DEVICES=0' in command
    assert str(output.parent.resolve()) + ':/results' in command
    assert '/results/run 01' in command and command[-1] == '--resume'


def test_ascend_explicit_mapping_and_read_only_toolkit(tmp_path):
    args = launcher.parse_args(['ascend9382', '--device', '14', '--card', '7', '--chip', '0',
                               '--suite', 'operators', '--output', str(tmp_path / 'run')])
    command = launcher.docker_command(args)
    assert '--privileged' in command and '--pid=host' in command
    assert 'ASCEND_RT_VISIBLE_DEVICES=14' in command
    assert 'FDR_NPU_SMI_CARD=7' in command and 'FDR_NPU_SMI_CHIP=0' in command
    assert '/usr/local/Ascend:/usr/local/Ascend:ro' in command
    assert command[-2:] == ['--suite', 'operators']


@pytest.mark.parametrize('options', [[], ['--device', '-1'], ['--figures-only', '--resume'],
                                    ['--figures-only', '--device', '0']])
def test_invalid_selection_is_rejected(options, tmp_path):
    with pytest.raises(SystemExit):
        launcher.parse_args(['a800', '--output', str(tmp_path / 'run'), *options])


def test_existing_output_is_preserved_and_docker_failure_propagates(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher.shutil, 'which', lambda _: '/usr/bin/docker')
    calls = []
    def execute(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=17)
    monkeypatch.setattr(launcher.subprocess, 'run', execute)
    output = tmp_path / 'run'
    output.mkdir()
    sentinel = output / 'keep.txt'
    sentinel.write_text('original data')
    with pytest.raises(SystemExit, match='already exists'):
        launcher.main(['a800', '--device', '0', '--output', str(output)])
    assert not calls and sentinel.read_text() == 'original data'
    assert launcher.main(['a800', '--device', '0', '--output', str(tmp_path / 'new')]) == 17
    assert calls[0][1] == {'check': False}
