# Ascend950 源码适配与已验收范围

更新：2026-10-29。实际Ascend950DT_9582上v8完整通过四算子六形状及KDA/GDN2独立消融。
公共autograd同步eager完整B组合几何均值：KDA R/A 1.2177/1.3031×，GDN R 1.2192×，
GDN2 R/A 2.4301/2.8945×，DeltaNet R 1.0878×；FB与独立贡献见正式报告，不混为旧低层计时。
详见[结果汇总](results.md)；调度和数值边界在下文说明。
源文件指纹用于防止运行错误版本，回归夹具保留实际 intra 输入。
本次公开整理只移除夹具元数据的内部路径，不改变内核或数值输入，
未进行硬件复测，也不将源码整理称为新的硬件验收。

## 实现

|算子|950 候选路径|调度与精度|
|---|---|---|
|KDA|现有 Triton residual WY + 修正版 FP32 矩阵 intra|分离调度；C32/K64；scalar beta 写两份 FP32 partial，再归约；无原生库|
|GDN2|现有完整片上 residual WY + 修正版 FP32 矩阵 intra|分离调度；C32/K64；vector beta 直接写最终db，无重排；原有 L/D 保持 FP32|
|GDN|现有统一伴随 + Gram 消除 wide 路径|保留统一 NK64/NV128 起始调度；950六形状及公共完整验收通过|
|DeltaNet|现有统一 chunk 伴随|保留既有精度与调度；950六形状及公共完整验收通过|

本轮完整前后向验收、240项六形状检查与120项零差异重复检查通过，范围限定于下述接口。
950实验preflight为KDA原始WY finalize/dw、GDN2原始WY finalize和DeltaNet residual producer
安装统一每program一个任务兼容调度，修复旧persistent路径重复性；方程与原精度保持。
另按真实248 KiB UB将原始GDN/GDN2 WY列块统一封顶64。冻结FLA磁盘源未修改。
复现必须执行该preflight；普通导入不会自动安装这些实验兼容路径。
未实现手写 950 RegBase/VF、Fixpipe 直传或跨核流水微内核。

`npu_target.py` 根据实际设备名选择后端。950 无原生 `.so` 依赖；旧原生
代码仅在 `research/operators/cannbot_npu_20261024/` 保留，供论文历史9382对照使用。
Ascend910_9382 于 10 月 29 日
另行修正为 FP32 Triton 路径，见 [9382 说明](ascend9382.zh-CN.md)；
未知设备拒绝自动冒充已审阅的平台。`FDR_EXPERIMENTAL_NPU=1` 仍是必要开关。

10 月 29 日回写：9382 副本复现了大 gate 跨度回退中的带三角掩码跨行加载错位，
950入口现复用两份已在9382验证的Triton数据流，读取合法行后显式掩码算术，
使用分块对角和唯一输出所有权。底层文件名含9382是验证来源，不是将设备改标；
入口仍强制核实物理950，核数来自当前设备。C32/K64已通过本次A5局部编译和数值回归，
不声称这是A5最优调度；固定六形状的完整长序列范围已经通过，不能外推到未测接口。

## 矩阵 intra 的数学与精度

设 `L=dAqk` 为含对角的下三角系数，`D=dAkk` 为严格下三角系数，`g` 为
冻结前向生成的 log2 chunk gate，`c` 是每列的中心值。

```
plus = 2**(g-c), minus = 2**(c-g)
u_q = (L @ (k*minus)) * plus
u_e = (D @ (k*minus)) * plus
u_k = (L.T @ (q*plus) + D.T @ (beta*k*plus)) * minus
dq += u_q
dk += beta*u_e + u_k
dg += q*u_q + k*(beta*u_e-u_k)
db += u_e*k                    # GDN2 channel erase
db += sum_K(u_e*k)             # KDA scalar beta
```

L/D、输入的 FP32 partial gradients、gate、所有新因子、dot 操作数和局部
累加均保持 FP32；`allow_tf32=False`。既有 residual 代码的局部 P 累加不变。
新增 KDA beta partial 是形状 `[K/64,B,T,H]` 的 FP32 scratch，不替换任何原有
FP32 张量；归约和分配均位于完整反向计时边界内。无新增前向保存。

原chunk仍C64，统一工作子块C32/K64，无case特化。每个对角子块gate列跨度小于80log2单位时使用
中心定标，单侧指数至多 `2**40`；超出时按原 pairwise gate 差做直接 FP32 求和。
该分支是必要的数值稳定分支，不切换原始实现，也不按 benchmark case 选择。
GDN2最终dq/dk保持原始BF16写出边界；已有FP32 partial及运算输入不降精度。
FP64 有限差分检查验证公式，不能代替 FP32 在真实 NPU 的数值与编译检查。

当前接口限定：BF16 q/k/v，K=V=128，等头数，固定长度，T 为 64 的倍数，
显式 initial_state / output_final_state，原有保存策略。支持原有 K/V state 布局转换；
packed/varlen、safe_gate、static-varlen graph、FP16 输入仍不在此首版范围内。

## 环境与执行

官方 [Triton-Ascend releases](https://github.com/triton-lang/triton-ascend/releases)
列有 950 支持；v3.2.1 公布的组合包括 CANN 9.0.0。实际主机必须记录其
CANN、torch_npu、Triton-Ascend 版本及真实编译 target，再选择兼容版本。
实测使用CANN9.2.0、PyTorch2.10.0、torch_npu2.10.0.post7、Triton-Ascend3.6.0；冻结FLA commit不变。

只有获得设备使用授权后才执行以下安装、探测、验收或计时命令：

```bash
bash scripts/install.sh npu950
python scripts/npu950_experiment.py check-source --expected-manifest validation/npu950_prepared_20261029.json --output results/950-source-01
python scripts/npu950_experiment.py preflight --expected-manifest validation/npu950_prepared_20261029.json --output results/950-preflight-01
python scripts/npu950_experiment.py verify --expected-manifest validation/npu950_prepared_20261029.json --output results/950-verify-01
python scripts/npu950_experiment.py benchmark --expected-manifest validation/npu950_prepared_20261029.json --output results/950-combined-01
# KDA/GDN2 分别保留 residual、矩阵 intra 和组合贡献：
python scripts/npu950_experiment.py benchmark --expected-manifest validation/npu950_prepared_20261029.json --ops kda,gdn2 --variant residual --output results/950-residual-01
python scripts/npu950_experiment.py benchmark --expected-manifest validation/npu950_prepared_20261029.json --ops kda,gdn2 --variant matrix --output results/950-matrix-01
```

CPU-only 计划生成可随时运行：

```bash
python scripts/npu950_experiment.py plan --output results/950-plan-01
```

每次输出要求新目录，记录物理型号、软件、源文件哈希、协议、完整验收和原始
样本；失败另存 `failure.json`。不覆盖旧证据。正式性能入口先执行 CPU FP64
参考验收，失败不计时；每规格另与冻结原始实现比较全梯度，检查实际优化命中，
禁止 fallback。更改 `FDR_NPU950_VARIANT` 须在一次完整实验开始前完成。

非plan入口强制核对预备源码清单；`check-source`只做CPU哈希检查，不导入Torch。
preflight还核对实际导入的950模块来自当前源码包，记录版本标识、真实核数和FLA
设备属性，并设置可写TEST_DATA_ROOT_PATH，防止UB查询目录错误沿用9382的失败。
verify/benchmark在公共验收前运行两家族真实FLA intra输入、门跨度和batch2回归，
对照独立CPU FP64收缩；失败不计时。输入来自9382实际调用，但在A5重新执行，
不会把原设备的通过状态直接当成A5通过。

首个 runner 采用 **公共 autograd 同步 eager B/FB**，含公共梯度转换、分配及
必要重算；不是旧论文低层图事件 B，也不是 STEP。固定六形状、30 轮对称
交替配对；保存策略分列。GDN/DeltaNet 公共入口没有 all-save 参数，因此
不伪造 A 策略；历史论文的实验缓存封装需另行迁移再补 A。独立 residual
和矩阵 intra 消融仅在 KDA/GDN2 开放，标量联合伴随不冒充独立贡献。
整网、图回放、未列出的尺寸/FP16接口验收及RegBase性能优化均不在已完成范围内。

## 历史精度更正

本次静态审计发现历史 NPU `gdn2_kernels.py::matrix` 同样把已有 FP32 L/D
及转置值转成 BF16。历史计时与数值报告保留，但独立/组合矩阵路径的严格
精度合规声明应撤下；见[精度标记](results.md#memory-components-and-display-selection)。新 950 路径不能追认旧结果，也
不把修复精度视为新独立代数贡献。
