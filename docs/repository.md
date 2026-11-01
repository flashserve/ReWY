# Repository Map

[Documentation](README.md) · [Contributing](../CONTRIBUTING.md)

| Location | Purpose |
| --- | --- |
| [`src/fast_delta_rule/`](../src/fast_delta_rule/) | Public API, routing, adapters and current kernels |
| [`examples/`](../examples/) | Runnable training example |
| [`tests/`](../tests/) | CPU tests and opt-in accelerator acceptance |
| [`tests/fixtures/`](../tests/fixtures/) | Minimal timing samples for aggregation regression |
| [`scripts/`](../scripts/) | Installation, Docker launcher, reproduction checks and NPU950 runner |
| [`benchmarks/`](../benchmarks/README.md) | Structured aggregates, six figure generators and algebra checks |
| [`research/`](../research/README.md) | Operator/model versions needed for reproduction and ablations |
| [`containers/`](../containers/README.md) | Published image pins and experiment build/run recipes |
| [`containers/runtime/`](../containers/runtime/README.md) | Current library images with the `fdr` entry point |
| [`validation/`](../validation/) | NPU matrix-input fixtures and runner source fingerprint |
| [`docs/`](README.md) | Installation, usage, compatibility, results and validation scope |

## Current Code and Reproduction Code

`src/` is the user-facing implementation. Dated directories under `research/`
retain the measured implementations and their dependencies. Historical
lower-precision controls stay explicitly labelled and isolated from public
default dispatch. They must not be presented as precision-qualified improvements.

`benchmarks/data/` contains numerical aggregates and plotting inputs, not the
manuscript or internal experiment archives. Figure generation reads JSON directly
and writes new output into `results/`. Regression fixtures retain only the samples
needed to check aggregation and missing-sample rejection.

## Source and Output Boundaries

This source distribution excludes the paper directory, collaboration records,
source-origin ledgers, failed deployment logs, machine inventories, credentials,
model weights, prebuilt device libraries and caches. It retains the project MIT
license and the complete FLA notice in [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

New experiment outputs go to an ignored `results/<run-id>/` directory. The
published images have immutable digests in [images.json](../containers/images.json)
and retain their own experiment snapshot. Rebuilding a current library or
experiment image does not replace those images.

The wheel contains only `fast_delta_rule`, its baseline/suite JSON files and
license notices. Source distributions use Git-tracked public files, including
the complete tests, fixtures and reproduction tools, without `MANIFEST.in`.

The NPU950 fingerprint checks the exact published runner inputs. Its fixture
metadata has no internal capture paths; kernel code and numerical fixture values
are unchanged by source publication. It is not a new hardware acceptance.
