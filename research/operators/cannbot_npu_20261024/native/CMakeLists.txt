cmake_minimum_required(VERSION 3.20)
find_package(ASC REQUIRED)
project(kda_residual_native LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(Torch REQUIRED PATHS "${TORCH_PREFIX}/Torch" NO_DEFAULT_PATH)
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import importlib.util,os; print(os.path.dirname(importlib.util.find_spec('torch_npu').origin))"
    OUTPUT_VARIABLE TORCH_NPU_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE)
set(ASCROOT "$ENV{ASCEND_HOME_PATH}/aarch64-linux/asc")
add_library(kda_residual_native SHARED residual_kernel.asc intra_kernel.asc extension.cpp)
set(KDA_INTRA_M 16 CACHE STRING "Rows owned by an intra AIV task")
set(KDA_INTRA_N 32 CACHE STRING "Key features owned by an intra AIV task")
set(KDA_INTRA_GLOBAL_ANCHOR 0 CACHE STRING "Use one whole-tile gate anchor")
target_compile_definitions(kda_residual_native PRIVATE KDA_INTRA_M=${KDA_INTRA_M} KDA_INTRA_N=${KDA_INTRA_N} KDA_INTRA_GLOBAL_ANCHOR=${KDA_INTRA_GLOBAL_ANCHOR})
target_include_directories(kda_residual_native PRIVATE
    ${TORCH_INCLUDE_DIRS} ${TORCH_NPU_ROOT}/include
    "$ENV{ASCEND_HOME_PATH}/include" ${ASCROOT}/include ${ASCROOT}/include/adv_api)
target_link_directories(kda_residual_native PRIVATE
    "$ENV{ASCEND_HOME_PATH}/lib64" ${TORCH_NPU_ROOT}/lib)
target_link_libraries(kda_residual_native PRIVATE ${TORCH_LIBRARIES} torch_npu tiling_api
    platform register ascendcl ascendc_runtime unified_dlog graph_base dl m)
target_compile_options(kda_residual_native PRIVATE ${TORCH_CXX_FLAGS}
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>)
