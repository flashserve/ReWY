# Contributing

Useful contributions include reproducible bug reports, clearer examples, model
integrations, device acceptance results and algorithmic improvements.

## Report an issue

Open a [GitHub issue](https://github.com/flashserve/ReWY/issues) with:

- A minimal command/script and the observed failure or result.
- `fdr doctor` output, device model and driver/toolkit versions.
- Operator, shape, dtype, save policy and selected integration.
- Dispatch report and error log, with credentials/private paths removed.

Performance reports should state B/FB/full-model timing, baseline, warmup,
repetitions and device occupancy. Include raw paired samples; a single best
timing cannot establish a speedup.

## CPU-only development

From a fresh Python 3.12 environment in the checkout:

```bash
python -m pip install -e '.[test]'
python -m pytest -q -m 'not accelerator'
python scripts/check_docs.py
python scripts/check_reproduction.py
python -m build
```

The package has no mandatory Torch dependency. These checks do not launch
accelerator experiments. [Figure dependencies](benchmarks/requirements.txt) are
needed separately for rendering and NumPy-based algebra checks.

## Make a change

Keep the public API in `src/fast_delta_rule/`, guides in `docs/`, and measured
research protocols under `research/`. See [the repository map](docs/repository.md)
and [architecture](docs/architecture.md). Keep both homepages consistent.

Preserve original FP32 matrices, accumulators, saving semantics and timing
boundaries. Avoid per-benchmark-case implementation selection. Historical
precision controls stay labelled and separate from current default kernels.

Kernel/adapter changes need relevant CPU checks and explicitly authorized device
acceptance. Documentation changes do not require a new performance campaign.
Do not regenerate source fingerprints to hide an unexplained kernel change.
Changed measured implementations need new validation records. Preserve the
950 prepared-source fingerprint unless the source changes are explicitly reviewed.

Pull requests should explain the problem, resulting behavior and checks run.
Hardware/model claims should name the tested scope and include raw evidence.
[Validation history](docs/validation.md) distinguishes routing, correctness,
performance and full-model diagnostics.
