"""Locations for public benchmark inputs and local check outputs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def check_output(name):
    directory = ROOT.parent / "build" / "checks"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name
