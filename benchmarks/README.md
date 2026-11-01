# Benchmarks and Figure Data

[Reproduction guide](../docs/reproduction.md) · [Results](../docs/results.md)

This directory contains the public numerical aggregates and CPU plotting tools
for the six ReWY figures. It does not contain the manuscript, internal experiment
archives, machine inventories or collaboration notes.

| Location | Purpose |
| --- | --- |
| `data/` | Operator/model aggregates, memory values, component tables and graph definitions |
| `plotting/` | Figure generators and independent NumPy algebra checks |
| `requirements.txt` | Pinned CPU plotting dependencies |
| `../research/` | Historical operator/model implementations and explicit experiment launchers |
| `../tests/fixtures/` | Minimal timing samples and regression inputs used by tests |

## Render the Figures

Run from the repository root:

```bash
python -m pip install -r benchmarks/requirements.txt
python benchmarks/plotting/make_figures.py --output-dir results/figures
python scripts/check_reproduction.py --render
python benchmarks/plotting/check_manuscript_math.py
python benchmarks/plotting/check_operator_identities.py
```

The generators read structured JSON, not LaTeX sources. New figures and numerical
snapshots go to `results/figures` by default, never back into the input data.
Numerical comparisons are exact; font and rasterizer differences can change
image bytes. Checks do not initialize an accelerator.

## Measurement Boundaries

The historical GPU/910C figures use complete low-level backward timing.
Ascend950 uses synchronized eager public-autograd backward. The current
`fdr benchmark` and `fdr reproduce` commands use the public boundary and are not
byte-identical replays of historical timings.

Historical lower-precision controls retain their dagger markers and are not
precision-qualified claims. Model backward gains exclude forward, optimizer
and data loading. New measurements belong in a new `results/<run-id>/` directory.
The published Docker images preserve the corresponding experiment environments;
their embedded source and archived data are separate from these aggregate inputs.
