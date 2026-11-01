#!/bin/bash
set -euo pipefail
export HOME="${HOME:-/tmp}" USER="${USER:-paper}" LOGNAME="${LOGNAME:-paper}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/fdr-triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/fdr-inductor}"
if [ "$FDR_PAPER_PLATFORM" = ascend9382 ]; then
    if [ "${1:-plan}" = run ]; then
        test -f /usr/local/Ascend/ascend-toolkit/set_env.sh || {
            echo 'Mount the recorded CANN 9.0.0-beta.1 installation at /usr/local/Ascend (read-only).' >&2
            exit 2
        }
        set +u
        source /usr/local/Ascend/ascend-toolkit/set_env.sh
        set -u
        : "${FDR_NPU_SMI_CARD:?Select the physical npu-smi card for telemetry}"
        : "${FDR_NPU_SMI_CHIP:?Select the physical npu-smi chip for telemetry}"
    fi
    export ASCEND_WORK_PATH="${ASCEND_WORK_PATH:-/tmp/fdr-ascend}"
    export TEST_DATA_ROOT_PATH="${TEST_DATA_ROOT_PATH:-/tmp/fdr-test-data}"
    export PATH="/tmp/zhangyi_kda_residual_wy_20261023/torch-env/bin:/usr/local/sbin:/usr/local/bin:$PATH"
    export PYTHONPATH="/opt/fast-delta-rule/src:/opt/model-deps:/opt/plot-deps:/opt/fla${PYTHONPATH:+:$PYTHONPATH}"
    unset FLA_DISABLE_BACKEND_DISPATCH
else
    export PATH="/workspace/env/bin:$PATH"
    export PYTHONPATH="/opt/fast-delta-rule/src:/opt/model-deps:/opt/fla${PYTHONPATH:+:$PYTHONPATH}"
    export LD_LIBRARY_PATH="/workspace/compat/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec python /opt/fast-delta-rule/containers/paper_suite.py "${@:-plan}"
