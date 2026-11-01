# ReWY documentation

[English home](../README.md) · [中文首页](../README.zh-CN.md)

## Use the library

1. [Install](installation.md): choose CUDA or Ascend and verify frozen FLA.
2. [Run your code](usage.md): launcher, Python API, model adapters and reports.
3. [Check compatibility](compatibility.md): supported inputs, precision and devices.
4. [Measure your workload](reproduction.md): public API versus paper protocols.

## Reproduce or study the paper

| Task | Guide |
| --- | --- |
| Inspect H100/H200 statefix1 results | [Results](results.md#h100-and-h200-operators) · [Image](../containers/hopper-statefix/README.zh-CN.md) |
| Rerun A800 or 9382 experiments | [Published Docker images](../containers/README.md) |
| Redraw all six figures on CPU | [Figures and data](../benchmarks/README.md#render-the-figures) |
| Understand the reported speedups and memory | [Results](results.md) |
| Find a kernel, ablation or model recipe | [Research map](../research/README.md) |
| Inspect the separate NPU studies | [9382 correction](ascend9382.zh-CN.md) · [950](ascend950.zh-CN.md) |

## Develop and review

[Repository map](repository.md) · [Architecture](architecture.md) ·
[Contributing](../CONTRIBUTING.md) · [Validation history](validation.md)

Start with the public package or experiment launcher. The dated research
directories preserve measured implementations; they are not an installation sequence.
