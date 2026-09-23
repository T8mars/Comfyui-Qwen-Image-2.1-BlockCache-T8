# Comfyui-Qwen-Image-2.1-BlockCache-T8

简体中文 | [English](README_EN.md)

面向 **ComfyUI 原生 Qwen-Image-2.1** 的四个独立 MODEL 节点：Block Cache、Spectrum、Sage Attention 和 Sol Attention，版本 `0.1.4`。使用官方模型加载器、文本条件、采样器和 VAE，不使用 Diffusers 包装管道。

`0.1.4` 新增按原生进度划分的前后两段阈值、可选 Sage/Kitchen 混合调度，以及两份已实测的混合编辑画布。旧工作流默认仍是固定阈值和原 Sage；混合模式未测得超过 Kitchen 单用的收益，不宣传叠加提速。

`0.1.3` 修复图像编辑反向加速：Block/Spectrum 现在保留官方参考图和文本的前缀 KV 缓存，串接 Spectrum 不再压低 Block 的连续命中上限。真实 7B INT8、1MP 人像编辑、40 步画布对照（Core compiler 开启）：Sage 基线 **34.43 秒 → 保守组合 24.74 秒**，约减少 28% 采样时间；Kitchen 基线 **33.56 秒 → 24.54 秒**。编辑示例使用 Block `0.03`、Spectrum `0.08`，而非较激进的文生图默认值。

当前仍为实验版，已验证的素材和 seed 有限，近似跳层会改变细节。**Sol 的 2048 完整模型测试期间发生系统重启，原因未定；Sol 默认关闭。** 1MP 编辑中强制启用 Sol 没有带来额外收益，勿把所有节点同时打开当作最快配置。[测试条件与画质差异](BENCHMARKS.md#image-edit-fix-013)。

## 可直接拖入画布的工作流

- [1024 文生图：Kitchen + Block Cache](workflows/Qwen21_T8_1024_T2I.json)
- [1024 文生图：Kitchen + Spectrum](workflows/Qwen21_T8_1024_Spectrum.json)
- [1MP 图像编辑：Sage + 保守 Block/Spectrum](workflows/Qwen21_T8_1024_Edit.json)
- [新增：1MP 图像编辑：混合后端 + Block/Spectrum](workflows/Qwen21_T8_1024_Hybrid_Edit.json)
- [新增：1MP 图像编辑：仅混合后端，不跳层](workflows/Qwen21_T8_1024_Hybrid_NoCache_Edit.json)

下载原始 JSON 后拖入 ComfyUI 画布，选择本机模型文件，再点击运行。它们是包含布局和连接的**前端工作流，不是 API JSON**；均已通过真实浏览器前端导入、点击运行和出图验证。紫色节点为旁路，选中后 `Ctrl+B` 切换；Sol 另外保持 `enabled=false`。

两份新增 JSON 均在本仓库的 `workflows` 文件夹，不在测试记录目录。保持实测采样设置，仅整理注释、布局及输出文件名前缀；混合组合示例仍用固定阈值 `constant`，需分段时手动选择 `two_stage`。上传自己的参考图，照片不随包分发。

另有 [1024 Sol 实验工作流](workflows/Qwen21_T8_1024_Sol.json)，已实际画布运行 25 步。关闭 Core compiler、增加显存预留后，串行对照 Kitchen 平均 16.61秒 → Sol 15.96秒，约快 4%；但出现额外乱码小字及壶盖细节改变。此图明确开启 Sol（`enabled=true, min_tokens=4096`），不是默认推荐配置；[完整条件与限制](BENCHMARKS.md#serial-1024-sol-follow-up--1024-串行复测)。不能据此认定 2048 安全。

历史文生图基准（0.1.1，4060 Ti 16GB、1024²、25步）：原生采样约 **18–19秒**；Block Cache 约 **11.3秒**；Kitchen + Block 约 **9.6秒**；Spectrum 约 **13.3秒**。不是 0.1.3 重测结果。本文速度仅统计采样节点；请严格串行测试，当前暂停 2048 压力测试。

编辑必须把 VAE 同时接入 `TextEncodeQwenImage21` 和解码节点；只接参考 IMAGE、漏接编码节点的 VAE，会缺少参考图 VAE 潜变量。需要保持原尺寸构图时，采样器连接该编码节点的 LATENT 输出；编辑示例按用户案例使用竖图尺寸，允许重新构图。参考照片不随插件分发，请选择自己的图。

## 安装与连接

在 ComfyUI 节点管理器搜索 **Qwen Image 2.1 BlockCache T8** 或 `qwen-image-21-blockcache-t8`，核对 Publisher 为 `t8star`。发布状态以 [Comfy Registry 页面](https://registry.comfy.org/publishers/t8star/nodes/qwen-image-21-blockcache-t8) 为准；若尚未索引，可刷新列表或手动安装：

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-Qwen-Image-2.1-BlockCache-T8.git
```

完整重启后在 `T8/Qwen Image 2.1` 分类添加节点。需要支持 Qwen2.1 的 ComfyUI（基线 0.37.0）及随 Core 配套的 Comfy Kitchen。Sage 节点另外要求当前 ComfyUI Python 环境已安装兼容的 SageAttention；本项目不会自动安装或下载任何依赖/模型。

已安装旧名 `Comfyui-Qwen-Image-2.1-T8` 的用户不要再装一份。在原插件目录更新远端并拉取即可；文件夹可保留旧名，四个节点的内部 ID 不变：

```bash
git remote set-url origin https://github.com/T8mars/Comfyui-Qwen-Image-2.1-BlockCache-T8.git
git pull --ff-only
```

继续使用 [官方 Qwen2.1 工作流](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_t2i.json)。将加速节点插在 MODEL 路径上，最后的 MODEL 接原生 sampler/guider/scheduler。独立 sampler 设置、提示词、参考图、透明通道和 VAE 流程保持官方方式。

```text
官方 Load Diffusion Model → 可选静态 LoRA
  → KJ 通用 Sage / T8 Sage / 官方 Model Attention Backend（按需选一个 dense 后端）
  → 可选 T8 Sol Attention
  → 可选 T8 Block Cache
  → 可选 T8 Spectrum
  → 官方采样路径
```

Comfy Kitchen 使用官方 `Model Attention Backend` 节点选择 `comfy kitchen attention`，不另造一个 Kitchen 节点。KJ 指通用 `Patch Sage Attention KJ`，不是 MiniMax H3 专用 Mem Eff Patch。已有外部 Sol 节点可保留在上游作为后端，但它们通常拒绝 Qwen2.1 的矩形 Q/K；实际 Sol 计算由本项目专用节点承担。

## 四个节点

| 节点 | 行为 | 实验默认值 |
| --- | --- | --- |
| Qwen Image 2.1 Block Cache (T8) | 每次真实计算 Block 0；指标稳定时复用目标图像的后续层残差，跳过 Block 1–31 | threshold `0.08`，范围 `0.10–0.85`，最多连续命中 `2` |
| Qwen Image 2.1 Spectrum (T8) | 用真实完整步的目标残差做 Chebyshev/ridge 预测；仍真实计算 Block 0 并进行变化检查 | history `4`，degree `2`，ridge `0.01`，guard `0.25`，范围 `0.15–0.85`，最多连续预测 `1` |
| Qwen Image 2.1 Sage Attention (T8) | 调用 Core 的 Sage 适配，保留矩形 Q/K、mask 和原生回退行为 | 无额外参数 |
| Qwen Image 2.1 Sol Attention (T8) | 实验性稀疏注意力；1024 限定条件测试通过，2048 重启原因未明，默认不执行 | enabled `false`，tau `1.0`，min_tokens `12288`，范围 `0.15–0.85` |

Block Cache 与 T8 Spectrum 可各自使用，也可前后串接：Block Cache 优先，未命中时再尝试预测；只将真实完整计算写入历史。每种算法用自己的上限检查自上次完整计算以来的连续跳过次数，Spectrum 不再把 Block 上限从 2 压成 1。缓存位置有任一选择 CPU 就用 CPU；内存预算取较小值。这些规则与连接顺序无关。组合收益取决于实际命中，`spectrum=0` 表示没有预测收益。

`cache_device=cpu` 为默认，`max_cache_mb=1024` 限制保留的缓存，超限淘汰旧流；运行时临时 anchor、重建输出和模型内存不计入该预算。bf16、2048² 输出时，一个目标 hidden 残差约 128 MiB，4 条 Spectrum 历史约 512 MiB/条件流；CFG 双流可能翻倍。预测按 512 tokens 分块 FP32 累加，避免构造巨大的特征系数矩阵。

`residual_diff_threshold`/`guard_threshold` 越高越容易跳层，但也更可能改变图像；`metric_stride` 越小检查越密、开销越高。Sol 的 `tau` 越高越激进。所有采样百分比使用原生 `percent_to_sigma`；首段/末段之外完整计算。

## Qwen2.1 专用设计和边界

### Sage / Kitchen 混合调度（0.1.4）

现有 Sage 节点新增可选 `backend_mode=sage_kitchen`：无 mask、至少1024个 query tokens、head_dim=128 的低精度 CUDA 注意力使用官方 Kitchen，其余保持 Sage 的原生适配路径；不支持 Kitchen 时回到 Sage。默认 `sage` 不改变旧工作流。混合模式不需要再串接官方 Kitchen 选择节点，仍可在后面连接 Block/Spectrum。

这是按调用形状分工，不是同一次 Attention 跑两遍，也不放宽跳层阈值。短文本/带 mask 调用由 Sage 适配器处理，当前 Sage 不支持的 mask 会按 Core 原有规则回退 PyTorch；终端 `dense routes` 统计的是适配器路由次数，不代表全部执行了 Sage kernel。Core 已有的 Kitchen RMS/RoPE、AdaLN、SwiGLU 融合仍然保留。

这是实验选项，不保证比 Kitchen 单用更快。切换量化后端可能改变图像，不能保证同 seed 像素一致；两种后端本身的收益不能相加。

本机1MP编辑、40步真实画布对照：Sage `35.018/34.742秒`，混合 `34.280/34.286秒`，Kitchen单用 `34.269秒`。混合只比该组Sage平均快约1.7%，没有超过Kitchen；与Block/Spectrum联用也已出图。详见[实测条件与限制](BENCHMARKS.md#attention-routing-014)。

### 两段阈值（0.1.4）

Block Cache 和 Spectrum 都支持 `threshold_mode=two_stage`。**所有边界均按原生采样进度，通过 `percent_to_sigma` 转换，不按步数或模型调用次数计数。** 原阈值分别作为前段阈值，新增 `late_threshold` 作为后段阈值，`split_ratio` 表示允许跳层区间内的前段占比。

```text
切换进度 = start_percent + (end_percent - start_percent) × split_ratio
```

例如两节点均设置 start=`0.15`、end=`0.85`、split_ratio=`0.5`：进度 `<0.15` 完整计算；`0.15≤进度<0.50` 使用各自原阈值；`0.50≤进度<0.85` 使用各自后段阈值；`≥0.85` 完整计算。比例改成 `0.3`，切换点就是 `0.36`，不是全程的30%。这不保证每段实际命中多少次，历史数量、变化检查和连续命中上限仍生效。

- Block 前段使用 `residual_diff_threshold`，后段使用 `late_threshold`。
- Spectrum 前段使用 `guard_threshold`，后段使用 `late_threshold`。
- 两节点可分别设置区间、比例和阈值；如果希望整个组合的首尾都完整计算，应让两节点首尾区间一致。把某一节点后段阈值设为0，只关闭该节点的后段跳层，另一个仍可能命中。
- `split_ratio=0` 全区间用后段阈值，`1` 全区间用前段阈值。默认 `constant` 忽略分段参数，保持旧工作流行为；新输入追加在原参数后，重启后重新载入旧画布即可出现。

分段功能经过 CPU 小模型和前端导入/保存/提示图参数检查；0.1.3 和 0.1.4 的真实大模型速度属于固定阈值测试，不作为两段配置的新性能结论。

### 模型边界

- 只识别原生 `QwenImage21Transformer2DModel`。不把旧 Qwen-Image、2512、H3 当成同一模型。
- Qwen2.1 是 32 层单流结构，只缓存目标图像 tail residual；文本/参考图不会作为待重建输出缓存。完整步仍走官方融合 RMS/RoPE、AdaLN、SwiGLU、量化线性层和 offload。
- 缓存仅属于一次采样，按条件 UUID、CFG 分支、shape、dtype、device、参考图布局区分。未知 UUID/sigma 不复用；sigma 重复或反向时刷新。正常完成、异常或取消均释放缓存。
- **通过 ModelPatcher 包装首尾块，保留原生 prefix KV 缓存。** 参考/文本单独计算时不参与跳层；目标图像路径才复用残差。无需改动 Core 文件；补丁按 ModelPatcher 生命周期恢复。官方 `Qwen Image 2.1 Cache` 的设备和精度设置仍由 Core 处理。
- 普通静态 LoRA 继续由原生加载器处理；存在 scheduled LoRA 的 `hook_patches` 时，Block/Spectrum 自动完整计算，防止跨权重缓存。
- Sol 将目标矩形 Q 前补查询后调用共享 kernel，K/V 不增删；丢弃补齐查询的输出，强制 prefix KV 和混合 64-token query 块精确计算。mask、参考图段、短序列、FP32、无内核或禁用低精度时保留原后端。
- Sol 需明确设置 `enabled=true` 才会尝试调用；1024² 约 4096 个目标 tokens，还会低于默认门槛而走 dense。2048 测试中断，不能据此宣称稳定或提速，不建议开启。
- 外部 Spectrum、EasyCache/LazyCache、替换/修改 block 的其他插件没有共享缓存协议，遇到这些已识别冲突会报出明确错误。需要组合 Spectrum 时用 **本仓库的 T8 Spectrum**。未承诺所有第三方 monkeypatch 都可自动识别。
- 同 seed 下，近似缓存、Spectrum、Sol 可能改变结果。CFG 拼批方式、调度器、分辨率和量化均影响命中率与质量。

终端汇总：`full=N cache=N spectrum=N peak cache=N MiB`；Sol 汇总：`kernel=N dense=N`。`kernel=0` 就没有调用 Sol kernel，不把回退称为加速。

## 验证

0.1.4：59 项 CPU/静态测试、4 项浏览器生命周期模拟通过，覆盖进度分段、后端路由及两份发布工作流；6次真实图像编辑画布测试全部串行完成。两份混合示例使用已出图的采样配置，未宣传两段阈值的新速度结果。

```powershell
python -m unittest discover -s tests -v
ruff check .
python -m compileall -q __init__.py nodes.py cache.py runtime.py attention.py tests
# 可选：仅小张量 GPU 探针，需能导入 ComfyUI（从根目录运行或配置 PYTHONPATH）
python tests/gpu_smoke.py
```

0.1.3：43 项 CPU/静态测试通过，新增带真实 Core 前缀 KV 的跳层、补丁恢复、无跳步 batch=2 数值等价、组合命中上限及编辑画布 VAE 接线回归。真实 7B INT8 编辑工作流已在浏览器画布导入、点击运行并保存 PNG，覆盖 Block、Spectrum、组合和实际调用 Sol 的组合；详见 [BENCHMARKS.md](BENCHMARKS.md)。

历史 0.1.2 审查：Core `e638023d`，Comfy Kitchen `0.2.35`，Torch `2.10.0+cu130`，39 项 CPU/静态测试串行运行 50 轮（1,950 次），另有 4 项浏览器生命周期模拟测试；不是 50 次大模型验证。

已保留低精度残差/重建溢出回退、混合 sigma 不启用 Sol、异常释放和串行测试锁。编辑新基准包含这些检查；早期文生图计时仍保留其历史来源。

RTX 4060 Ti 小张量探针：BF16/FP16 Sol compiled-vs-eager 相对 L2 约 `0.01225/0.01234`；这是数值测试，不代表完整模型 Sol 稳定。真实 INT8 7B、原生 DynamicVRAM、512/1024 采样与画布验证见 [BENCHMARKS.md](BENCHMARKS.md)。修复了 Spectrum 默认二次外推被旧限制全部拒绝的问题；1024 实测预测命中 9/25 次，Block 命中 11/25 次。

## 许可与发布状态

插件自写代码采用 [Apache-2.0](LICENSE)，研究和归属见 [NOTICE](NOTICE.md)。Qwen 模型材料遵循其独立许可证；不随本包发布权重。Spectrum 节点是针对 Qwen2.1 的简化谱预测适配，不代表原论文完整控制策略。

代码仓库：[T8mars/Comfyui-Qwen-Image-2.1-BlockCache-T8](https://github.com/T8mars/Comfyui-Qwen-Image-2.1-BlockCache-T8)。Registry 包 ID：`qwen-image-21-blockcache-t8`；Publisher：`t8star`。采用 [ComfyUI 官方发布流程](https://docs.comfy.org/registry/publishing)，推送 `pyproject.toml` 更新后由 GitHub Actions 发布；GitHub 推送成功不等于 Registry 已完成处理，请查看 [发布任务](https://github.com/T8mars/Comfyui-Qwen-Image-2.1-BlockCache-T8/actions/workflows/publish_action.yml) 和 Registry 状态。

`0.1.3`：修复编辑时失去官方前缀 KV 缓存造成的减速；修复组合命中上限；减少残差 CPU 传输；新增实测编辑画布。研究记录、机器日志、模型、参考照片及生成图片不随包发布。
