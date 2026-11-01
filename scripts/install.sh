#!/usr/bin/env bash
set -euo pipefail
target="${1:-cuda}"
root="$(cd "$(dirname "$0")/.." && pwd)"
python_bin="${FDR_PYTHON:-python}"
revision=a59bc5df8c5841dce2169bb00e4027287197ba77
case "$target" in
  cuda)
    "$python_bin" -m pip install 'torch==2.9.1' --index-url https://download.pytorch.org/whl/cu128
    "$python_bin" -m pip install 'triton==3.5.1' 'einops==0.8.2'
    ;;
  npu|npu950)
    "$python_bin" -c 'import torch, torch_npu, triton; assert torch.npu.is_available(), "Activate your CANN / torch_npu / Triton-Ascend environment first"'
    ;;
  *) echo 'Usage: bash scripts/install.sh cuda|npu|npu950' >&2; exit 2 ;;
esac
"$python_bin" -m pip install 'transformers==5.21.0' 'einops==0.8.2' "git+https://github.com/fla-org/flash-linear-attention.git@$revision"
"$python_bin" -m pip install "$root"
# Current 9382/950 paths use FP32 Triton-Ascend intra; native historical
# implementations are isolated under research/operators for paper evidence.
if [[ "$target" == npu950 ]]; then
  "$python_bin" -c 'import torch, torch_npu; from fast_delta_rule.backends.npu_target import require_950; require_950()'
fi
"$python_bin" -m fast_delta_rule doctor --check-sources
