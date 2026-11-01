#!/usr/bin/env python3
"""Check public experiment plans, data and optional CPU figure rendering."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
FIGURES = (
    "intro_overview", "dataflow", "backward_speedups",
    "backward_memory_reduction", "component_speedups", "component_memory",
)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_rendered(output, manifest, strict_bytes=False):
    for name in ("figure_data.json", "intro_graph_data.json"):
        assert read_json(output / name) == read_json(BENCHMARKS / "data" / name), f"Figure values differ: {name}"
    differences = [
        name for name, expected in manifest["figure_png_sha256"].items()
        if digest((output / name).read_bytes()) != expected
    ]
    if strict_bytes:
        assert not differences, f"Figure bytes differ: {differences}"
    return differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Requires benchmarks/requirements.txt.")
    parser.add_argument("--strict-figure-bytes", action="store_true")
    args = parser.parse_args()
    if args.strict_figure_bytes and not args.render:
        parser.error("--strict-figure-bytes requires --render.")

    recipes = read_json(ROOT / "research/model_runs.json")
    cases = recipes["a800"] + recipes["npu9382"]
    assert len({row["case"] for row in cases}) == len(cases) == 21
    for row in cases:
        entry = ROOT / "research/models" / row["source_folder"] / row["entry"]
        tree = ast.parse(entry.read_text(encoding="utf-8"))
        options = {
            arg.value for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
        }
        assert {arg for arg in row["argv"] if arg.startswith("--")} <= options, row["case"]

    spec = importlib.util.spec_from_file_location("public_suite", ROOT / "containers/paper_suite.py")
    suite = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(suite)
    for platform in ("a800", "ascend9382"):
        for job in suite.plan(platform)["jobs"]:
            assert (ROOT / job["entry"]).is_file(), job["entry"]
    for name in FIGURES:
        assert (ROOT / "docs/figures" / (name + ".png")).is_file(), name
    for path in (BENCHMARKS / "data").glob("*.json"):
        read_json(path)
    for path in (ROOT / "src").rglob("*.py"):
        compile(path.read_bytes(), str(path), "exec")
    spec = importlib.util.spec_from_file_location("npu950_check", ROOT / "scripts/npu950_experiment.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    seal = runner.check_prepared(ROOT / "validation/npu950_prepared_20261029.json")

    differences = None
    if args.render:
        manifest = {"figure_png_sha256": {
            name + ".png": digest((ROOT / "docs/figures" / (name + ".png")).read_bytes())
            for name in FIGURES
        }}
        with tempfile.TemporaryDirectory(prefix="rewy-figures-") as folder:
            subprocess.run([
                sys.executable, str(BENCHMARKS / "plotting/make_figures.py"),
                "--output-dir", folder,
            ], check=True)
            differences = verify_rendered(Path(folder), manifest, args.strict_figure_bytes)
    print(json.dumps({
        "status": "PASS", "model_cases": len(cases), "figures": len(FIGURES),
        "npu950_source_files": seal["source_files"], "rendered": args.render,
        "figure_byte_differences": differences, "accelerator_execution": False,
    }, indent=2))


if __name__ == "__main__":
    main()
