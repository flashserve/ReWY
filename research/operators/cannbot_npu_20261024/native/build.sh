#!/usr/bin/env bash
set -e
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
task_python="${KDA_PYTHON:-python}"
task_dir="$(cd "$(dirname "$0")" && pwd)"
cmake -S "$task_dir" -B "$task_dir/build" -DCMAKE_BUILD_TYPE=Release \
    -DPython3_EXECUTABLE="$task_python" -DCMAKE_PREFIX_PATH="$ASCEND_HOME_PATH/aarch64-linux/lib64/cmake" \
    -DKDA_INTRA_M="${KDA_INTRA_M:-16}" -DKDA_INTRA_N="${KDA_INTRA_N:-32}" \
    -DKDA_INTRA_GLOBAL_ANCHOR="${KDA_INTRA_GLOBAL_ANCHOR:-0}"
cmake --build "$task_dir/build" -j4
