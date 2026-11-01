# H100/H200 D256 状态反向修复

镜像：`zhangyi2026/residual-wy-h100:20261031-cu126-triton371-statefix1`

摘要：`sha256:f70af7071d1b8c4aa27d5055cb9f7836d568ba0cfb0fdc223e4244cbaa115633`

```bash
docker pull zhangyi2026/residual-wy-h100:20261031-cu126-triton371-statefix1
docker run --rm zhangyi2026/residual-wy-h100:20261031-cu126-triton371-statefix1 verify
```

`syncmma1` 的 DeltaNet 优化状态核在 D256 上需要 315,392 字节共享内存，超过 H100/H200 的 232,448 字节上限。
本补丁仅给该优化入口添加 `arch='sm80'`，使用同步 MMA 降低路径；最终 PTX/cubin 仍针对实际 SM90。
四 warps、三阶段、BV16、冻结 JIT 运算正文、FP32 累加与保存策略保持，原始对照继续使用原编译路径。
所有形状使用同一调度，没有 D256 特化、降精度或全局编译器覆盖。

修复后真实入口的 D256 资源需求为 **215,040 字节**，余量 17,408 字节。
旧检查手写 ASTSource 的属性没有复现真实启动的指针对齐信息，错误报告 20,480 字节；
新检查通过实际 `Operator.backward` 调用和 JIT warmup 构造参数，保留 315,392 字节超限负对照。

验收：20 项容器测试、4 个入口检查、41 项 SM90 资源正例通过；37 项受影响状态配置在镜像的 Triton 3.7.1 中验证 SM80 新旧 cubin 完全一致。
A800_1 的 `zy_delta_rule`（PyTorch 2.9.1 / Triton 3.5.1）完成六规格 R/A 的 24 组 CPU FP64 梯度检查、12 组 Graph 回放和 memcheck，0 errors。
D256 的 215,040 字节超过 A800 的 166,912 字节上限，因此它的验收是 SM90 离线实际启动编译；镜像保留完整七规格 Hopper 检查。
**2026-11-01 更新：H100/H200 的 10 月 31 日 statefix1 实测已纳入论文。**
每卡四算子、两种保存策略、六形状共 48 配置，另有 45 个整网配置。
算子完整低层 B 六形状几何均值为：

| 算子 | H100 R / A | H200 R / A |
| --- | ---: | ---: |
| DeltaNet | 1.2066 / 1.3046× | 1.1776 / 1.2697× |
| GDN | 1.2056 / 1.2829× | 1.1819 / 1.2505× |
| KDA | 1.3264 / 1.4213× | 1.3065 / 1.3964× |
| GDN2 | 1.5680 / 1.7033× | 1.5408 / 1.6527× |

原始设备名为 NVIDIA H100 80GB HBM3 和 NVIDIA H200。嵌入式算子 CPU FP64、
完整梯度及 Graph 检查通过；额外 focused sanitizer 任务跳过，不能宣称本版全面 Hopper
Sanitizer 验收。正式性能网格为 D128，不能将这些数值称为 D256 实测收益。
整网仅统计纯 B，保留数值诊断限制与结构探索分类，详见
[结果与边界](../../docs/results.md)、[公开聚合数据](../../benchmarks/data/h100_data.json)。
构建与资源检查不等同于上述性能测量。

在获授权的 H100/H200 上复测报错用例（显式指定空闲 GPU UUID）：

```bash
IMAGE=zhangyi2026/residual-wy-h100:20261031-cu126-triton371-statefix1
GPU=GPU-替换为完整UUID
mkdir -p results-statefix
docker run --rm --gpus "device=$GPU" --shm-size=16g \
  -e CUDA_VISIBLE_DEVICES="$GPU" -e BENCH_GPU="$GPU" \
  -v "$PWD/results-statefix:/results" --entrypoint bash "$IMAGE" -lc '
  set -euo pipefail
  compute-sanitizer --tool memcheck --target-processes all --error-exitcode 99 \
    python check_gdn_runtime.py --gpu "$BENCH_GPU" --operator delta \
    --output /results/delta-regression.jsonl
  '
```

容器退出码应为 0，JSONL 末行应为 complete/passed，Sanitizer 应为 0 errors。
原标签和已有结果保留，镜像内既有单算子和整网运行入口保持。
整网统计仍只使用纯反向 B。

可复现构建：在本目录执行 `docker build -t zhangyi2026/residual-wy-h100:20261031-cu126-triton371-statefix1 .`。Dockerfile 固定父镜像摘要；构建不使用 GPU。
