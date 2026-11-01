from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_public_tree_excludes_internal_material():
    for name in ("paper", "backup", "MANIFEST.in", "provenance.json", "licenses",
                 "research/evidence", "containers/hopper-statefix/validation",
                 "validation/containers-20261030"):
        assert not (ROOT / name).exists(), name


def test_third_party_permission_is_preserved():
    text = (ROOT / "THIRD_PARTY_NOTICES.md").read_text()
    assert "Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li" in text
    assert "The above copyright notice and this permission notice" in text
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in text


def test_native_sources_have_local_headers():
    import re
    root=Path(__file__).resolve().parents[1]/'research/operators/cannbot_npu_20261024/native'
    for path in root.iterdir():
        if path.suffix not in ('.asc','.cpp','.h'):continue
        for header in re.findall(r'#include "([^/\"]+)"',path.read_text(encoding='utf8')):
            if header in {'kernel_operator.h','kernel_tiling.h'}: continue  # CANN SDK headers.
            assert (root/header).is_file(),(path,header)
