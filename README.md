# Comfyui-Qwen-Image-2.1-BlockCache-T8

简体中文 | [English](README_EN.md)

适用于 **ComfyUI 原生 Qwen-Image-2.1** 的实验性加速节点。沿用官方模型加载、条件编码、采样和 VAE 流程；只在 MODEL 连线上增加注意力后端或跳层缓存。

## 安装

需要支持 Qwen-Image-2.1 的 ComfyUI（最低 `0.37.0`）。在节点管理器搜索 `qwen-image-21-blockcache-t8`，确认 Publisher 为 `t8star`；或手动安装：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-Qwen-Image-2.1-BlockCache-T8.git
```

安装后重启 ComfyUI。T8 Sage 需要在同一 Python 环境安装兼容的 SageAttention；Kitchen 使用 ComfyUI 自带的 `Model Attention Backend` 节点。已装旧仓库名 `Comfyui-Qwen-Image-2.1-T8` 的用户请更新原目录的 Git 远端，不要重复安装。

## 画布工作流

以下都是可拖入 ComfyUI 的**前端画布 JSON**，不是 API 工作流。导入后选择本机模型；编辑工作流还需换成自己的参考图。

| 场景 | 工作流 |
| --- | --- |
| 文生图 | [Kitchen + Block Cache](workflows/Qwen21_T8_1024_T2I.json) · [Kitchen + Spectrum](workflows/Qwen21_T8_1024_Spectrum.json) |
| 图像编辑 | [Sage + Block/Spectrum](workflows/Qwen21_T8_1024_Edit.json) · [混合后端 + Block/Spectrum](workflows/Qwen21_T8_1024_Hybrid_Edit.json) · [仅混合后端，不跳层](workflows/Qwen21_T8_1024_Hybrid_NoCache_Edit.json) |
| 实验 | [Sol 文生图](workflows/Qwen21_T8_1024_Sol.json) |

图像编辑时，参考图及 VAE 都要接入 `TextEncodeQwenImage21`，VAE 还需接解码节点；漏接编码侧 VAE 会导致编辑结果失真。要保持原图画幅，请使用编码节点的 LATENT 输出。紫色旁路节点可选中后用 `Ctrl+B` 启用；Sol 另有独立的 `enabled` 开关。

## 节点怎么连接

```text
官方模型加载器 → 可选 LoRA → 注意力后端（任选一种）
  → 可选 T8 Block Cache → 可选 T8 Spectrum → 官方采样器
```

注意力后端可选官方 Kitchen、通用 KJ Sage 或 T8 Sage；**不要把多个后端的收益相加**。T8 Sage 的 `backend_mode=sage_kitchen` 是实验性的按调用形状路由，使用时无需再串接 Kitchen 节点；本机实测未快于单用 Kitchen。T8 Sol 默认关闭，不建议作为日常配置。

| 节点 | 用途 | 起步设置 |
| --- | --- | --- |
| Block Cache | 检查首块变化，稳定时复用后续块残差 | 图像编辑先试 `residual_diff_threshold=0.03` |
| Spectrum | 用历史完整计算预测残差；可与本仓库 Block Cache 串接 | 图像编辑先试 `guard_threshold=0.08` |
| Sage Attention | 原生 Sage 适配；可选 Sage/Kitchen 混合路由 | 默认 `backend_mode=sage` |
| Sol Attention | 实验性稀疏注意力 | 默认 `enabled=false` |

Block/Spectrum 的阈值越高，越容易跳过计算，也越可能改变人物、文字和细节。两节点都有 `start_percent`、`end_percent` 控制可跳层区间；区间外完整计算。`max_consecutive_hits` 限制连续跳过次数。建议先单独开启一个节点，对比不加速的输出，再组合使用。它们只支持原生 Qwen-Image-2.1，不与外部 EasyCache/Spectrum 共享缓存。

### 前后两段阈值

将 Block 或 Spectrum 的 `threshold_mode` 改为 `two_stage`：原阈值用于前段，`late_threshold` 用于后段，`split_ratio` 决定**可跳层区间内**前段所占比例。切换进度为：

```text
start_percent + (end_percent - start_percent) × split_ratio
```

例如 `start=0.15`、`end=0.85`、`split_ratio=0.5`，就在采样进度 `0.50` 切换阈值，**不是按步数切换**。默认 `constant` 保持旧工作流行为。两段模式已通过功能测试，下面的速度数据仍是固定阈值配置，不能当作两段模式跑分。

## 实测与限制

7B INT8、1MP 人像编辑、40 步、Core compiler 开启的画布采样：Sage 基线 **34.43 → 24.74 秒**，Kitchen 基线 **33.56 → 24.54 秒**（组合 Block `0.03` + Spectrum `0.08`）。这是有限素材与 seed 下的**采样时间**，不是整条工作流耗时或画质保证。混合 Sage/Kitchen 在该场景中与单用 Kitchen 基本持平；Sol 的 2048 测试曾导致系统重启，原因未明。详细条件、其他跑分及画质差异见 [BENCHMARKS.md](BENCHMARKS.md)。

代码采用 [Apache-2.0](LICENSE)；模型权重不随插件分发，来源与归属见 [NOTICE](NOTICE.md)。
