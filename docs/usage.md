# Use ReWY in your code

[Documentation](README.md) · [Installation](installation.md) · [Compatibility](compatibility.md)

## Existing training scripts

```bash
fdr run --backend original train.py --config config.yaml
fdr run --backend optimized --report results/dispatch.json train.py --config config.yaml
```

Arguments after the script name belong to your script. The launcher enables
the backend before that script imports FLA, without rewriting source files.
Use `--ops kda` or a comma-separated set before the script name to select
families. By default all four are eligible.

Try the included example before a larger workload:

```bash
fdr run --backend optimized --report results/example.json examples/train_step.py --steps 3
```

This is a small CUDA KDA training example, not a full-model benchmark.
`--fallback original` explicitly permits unsupported calls to use the original
implementation and reports why. Compilation/runtime failures still propagate.

## Python API

For an existing FLA model or notebook, enable the backend **before importing
model and operator modules**:

```python
import fast_delta_rule as fdr

fdr.enable(ops=("kda",))
from fla.ops.kda import chunk_kda

# Use chunk_kda in your existing forward/backward code.
```

Or call the package functions directly. Assuming `q`, `k`, `v`, `g` and `beta`
are your supported KDA tensors with gradients enabled:

```python
import fast_delta_rule as fdr

fdr.stats(reset=True)
with fdr.use_backend("optimized", ops=("kda",)):
    output, final_state = fdr.chunk_kda(q, k, v, g, beta)

output.float().square().mean().backward()
fdr.assert_used(("kda",))
print(fdr.stats())
```

Backward stays bound to the implementation selected during forward, even after
the context exits. Direct functions use the original backend outside an enabled
or optimized scope. Import alone does not patch FLA. `fdr.disable()` selects
original behavior for future forwards.

| Family | Direct function | CLI name |
| --- | --- | --- |
| KDA | `fdr.chunk_kda` | `kda` |
| Gated DeltaNet | `fdr.chunk_gated_delta_rule` | `gdn` |
| GDN2 | `fdr.chunk_gdn2` | `gdn2` |
| DeltaNet | `fdr.chunk_delta_rule` | `delta` |

Functions retain the frozen FLA signatures; the supported subset is in
[compatibility](compatibility.md). NPU calls require `FDR_EXPERIMENTAL_NPU=1`,
`initial_state` and `output_final_state=True`.

## Model adapters

FLA integration is enabled by default. Request additional adapters explicitly:

```bash
fdr run --integration qwen3_5 --ops gdn --report results/qwen.json train.py
fdr run --integration gdn2 --ops gdn2 --report results/gdn2.json train.py
```

`qwen3_5` binds the documented Transformers Qwen3.5 delta-rule entry. `gdn2`
binds `lit_gpt.gdn2.chunk_gdn2`, requiring the upstream project and dependencies
on Python's import path. Adapters select the delta-rule core, not a complete
model implementation. They do not certify arbitrary architectures, convergence,
distributed execution or `torch.compile`.

## Confirm the backend was used

A successful optimized training run has a positive `<op>.optimized_backward`
count in the dispatch report. The launcher raises an error when no selected
optimized backward ran. In Python, `fdr.assert_used(("kda",))` can require a
specific operator after backward.

If a counter is missing, check that:

1. Setup happened before importing model/operator aliases.
2. The model uses a supported FLA operator or the required adapter was selected.
3. The script ran backward and `--ops` includes that family.

`--allow-no-backward` is for deliberately forward-only work. Counters measure
Python submissions, not subsequent device-graph replays. A dispatch count
confirms routing; use [matched measurements](reproduction.md) for speedups.
