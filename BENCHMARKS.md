# 真实模型与画布验证 / Real-model and canvas validation

The T2I tables below are historical 0.1.1 measurements. Current image-edit results are in [0.1.3 edit validation](#image-edit-fix-013).

2026-09-23，RTX 4060 Ti 16 GB / Windows，ComfyUI `e638023d` (0.37.0)，Frontend 1.53.6，Torch 2.10.0+cu130，Kitchen 0.2.35，Aimdo 0.5.5。使用原生 DynamicVRAM，`--reserve-vram 3 --vram-headroom 1 --preview-method none`。不修改 Core，不下载模型。

Weights: `qwen_image_2.1_int8_convrot.safetensors` (7B), `qwen3vl_8b_fp8_scaled.safetensors`, `qwen_image_2.1_vae_bf16.safetensors`. Seed 42, CFG 1, Euler/simple, 25 steps, batch 1. Prompt is the red-teapot example in the canvas workflows. Cache storage is CPU. This is limited T2I evidence, not a guarantee across prompts, hardware, editing or model formats.

Timing provenance: the GPU measurements below predate the subsequent non-finite cache checks and mixed-sigma Sol guard. That follow-up passed 50 serial CPU regression rounds, not new full-model benchmarks. Added reductions may have a cost; unchanged GPU performance has not been established.

## 1024 × 1024

KSampler 的实测墙钟时间，包含该节点的加载/切换开销，不包含文本编码与 VAE/保存。每次实际采样，拒绝把执行缓存命中计时；同输入的重复原生基线像素一致。没有宣称统计显著性。

Measured KSampler wall time, including its model activation/switching overhead, excluding text encoding and VAE/save. Execution-cache hits are rejected. Repeated native baseline images were pixel-identical. These are small-sample observations, not statistically established performance guarantees.

| Configuration | Sampling seconds | Actual tail skips / 25 |
| --- | ---: | ---: |
| Native default (xformers), repeated | 18.13–19.23 | 0 |
| T8 Block Cache, default settings | 11.26 | 11 cached |
| T8 Spectrum, after forecast fix | 13.33 | 9 predicted |
| T8 Block + Spectrum | 11.61 | 10 cached, 0 predicted |
| Native Kitchen only | 15.76 | 0 |
| Kitchen + T8 Block, repeated | 9.55–9.65 | 11 cached |
| Kitchen + T8 Spectrum | 11.81 | 9 predicted |
| T8 Sage + T8 Block | 9.56 | 11 cached |

Example: Kitchen + Block reduces sampling time by about 47–50% versus this native default, or about 39% versus Kitchen alone. **This is not a promise of 2× total workflow speed.** Prefer Block alone when preserving the existing attention backend; replacing attention can itself change the image substantially. These historical combination measurements used the stricter shared consecutive-skip cap; version 0.1.3 fixes that behavior.

512 × 512 first-pass comparison: native 3.78–3.97 s; Block 2.55 s with 11 hits. The pre-fix Spectrum did not predict; those old timings are not used to claim Spectrum acceleration.

## Output differences / 输出差异

RGB pixel comparisons on the same prompt and seed; these measure similarity, **not general perceptual quality**. Images were also visually inspected: the teapot, card text, plant and scene remained present, with changes to detail/shape. This is not a broad quality regression suite.

| Compared to same-backend baseline | SSIM | PSNR (dB) |
| --- | ---: | ---: |
| Block vs native | 0.9867 | 35.53 |
| Spectrum vs native | 0.9623 | 26.09 |
| Kitchen + Block vs Kitchen | 0.9827 | 31.78 |
| Kitchen + Spectrum vs Kitchen | 0.9719 | 27.86 |

Kitchen + Block vs the different xformers backend gives SSIM 0.8983; do not attribute all backend differences to caching or advertise bit-exact output.

## Canvas verification / 画布验证

The files in `workflows/` are **frontend workflow JSON**, containing nodes, links, positions and widget values, not API prompt dictionaries. They were created/serialized by the real ComfyUI frontend, imported through its file input, and executed by clicking the visible Run button in Chromium. All completed sampling, VAE decoding and PNG saving. Returned PNG metadata contains the canvas workflow as well. No missing-node dialog or submission error occurred.

- `Qwen21_T8_1024_T2I.json`: Kitchen + Block active; Sage/Sol/Spectrum bypassed.
- `Qwen21_T8_1024_Spectrum.json`: Kitchen + Spectrum active; Block/Sage/Sol bypassed.
- `Qwen21_T8_1024_Sol.json`: experimental Kitchen + Sol test; Block/Sage/Spectrum bypassed. Requires the separate test conditions below.

The saved layout/notes were subsequently tidied without changing either tested active sampling path. Sol's additional `enabled=false` control is inactive in both workflows. Models must already exist locally; choose your filenames after import. Sage is optional for these two workflows.

## Sol limitation and test safety / Sol 限制与测试安全

Small BF16/FP16 compiled Sol kernel tests passed. A 2048 Kitchen baseline finished in 93.41 s; the following Kitchen + Sol run stopped at step 14/25 when the workstation restarted. No Python traceback establishes the cause. Do not infer a Sol speedup from partial step timings or claim the restart was definitively caused by a particular component.

Sol defaults to `enabled=false`; the Block and Spectrum workflows bypass it. Large-image testing is suspended. No 2048 workflow is distributed. **All future tests must be strictly serial**, one runner/workflow at a time. The API and canvas test scripts share an exclusive lock and refuse to enqueue when ComfyUI is busy. A stale lock after a crash must be reviewed and removed only after verifying no test is running. These guards do not impose a hard GPU-memory limit or guarantee driver stability.

### Serial 1024 Sol follow-up / 1024 串行复测

2026-09-23，用户指定 1024 后完成：5 步 Sol 检查、Kitchen 25 步、Sol 25 步、Kitchen 25 步复测、真实浏览器画布 Sol 25 步。每项结束卸载模型并释放缓存，再开始下一项。全部成功，无重启；结束后关闭专用测试服务。

Same hardware, weights, prompt, seed and sampler as above, but **different server flags**: `--reserve-vram 5 --vram-headroom 3 --disable-comfy-compiler --preview-method none`. Core compilation/CUDA graphs were disabled for this follow-up, not Sol's compiled kernel. Do not compare these timings directly with the earlier table or extrapolate to default compiler settings or 2048. Sol: `enabled=true`, `min_tokens=4096`, `tau=1.0`, window `0.15–0.85`; no Block/Spectrum. The node's production defaults remain unchanged.

| Configuration | KSampler seconds, two runs | Mean |
| --- | --- | ---: |
| Kitchen | 16.631 / 16.590 | 16.61 |
| Kitchen + Sol (second run from canvas) | 15.937 / 15.989 | 15.96 |

About **3.9% less sampler time (1.04×)** in this small sample. Each full Sol run logged `kernel=576 dense=256`, confirming actual sparse-kernel use. This is not a total-workflow speedup or broad stability guarantee.

Compared with the same-backend Kitchen baseline, both Sol images measured SSIM **0.9430**, PSNR **25.38 dB**, RGB MAE **0.01928** (0–1 scale). Visual inspection found an extra garbled text line on the card and changed teapot-lid geometry. **The modest speed gain has a visible quality tradeoff; Sol remains off by default.** These results do not establish which tau best balances speed and quality across prompts.

复现画布：`Qwen21_T8_1024_Sol.json` 已实际导入前端、点击运行并保存 PNG，不只是 API 提交。可选自动化命令：`node tests/canvas_smoke.cjs --sol --run`，需已有 Playwright/Chromium 和上述专用空闲服务；不能与其他测试同时运行。

## Reproduction

Use a dedicated idle localhost ComfyUI server. Start with 512, one mode per invocation:

```text
python tests/benchmark_native.py --size 512 --steps 25 --modes baseline
python tests/benchmark_native.py --size 512 --steps 25 --modes block
```

Unload models and release cached memory after each test before starting the next. Unknown/multiple modes are rejected. API, canvas and tiny GPU probes share `benchmark_results/run.lock`; if a submitted job's final state cannot be confirmed, API/canvas retain the lock. Verify the server has no running/pending work before manually removing a stale lock. Closing the browser does not cancel a server job. These locks cannot prevent unrelated applications or other users from using the GPU.

For frontend verification, import a file from `workflows/` and click Run. `tests/canvas_smoke.cjs --run` automates that with an existing Playwright installation; it is optional and not a runtime dependency. Do not run it while another benchmark is active. `tests/compare_outputs.py` optionally compares local PNGs with Pillow/NumPy/scikit-image. No models, test outputs, local research notes or machine logs are included in the repository.

## Image edit fix 0.1.3

2026-09-23. Same hardware/Core/model files as above. A single user-supplied portrait reference, prompt `女人吃苹果`, seed 42, CFG 1, Euler/simple, **40 steps**, batch 1. VAE is connected to TextEncodeQwenImage21. Reference resolution=1024 (1440×736 after native resizing); portrait target is 768×1376 after native latent sizing. This is an actual editing workflow, not the earlier teapot T2I benchmark. The reference/output photographs remain local.

Every row was imported into the real browser canvas, saved as a UI workflow, re-imported, and executed by clicking Run. `tests/canvas_edit.cjs` records submitted graphs, histories and websocket sampler durations. Tests were strictly serial; models were unloaded between runs. Flags for the first table: `--use-sage-attention --cuda-malloc --disable-smart-memory --reserve-vram 5 --vram-headroom 3 --disable-comfy-compiler --preview-method none`. CPU residual storage. The same effective Sage backend is used in every row. Do not compare these times directly with other hardware or the historical T2I table.

| Configuration | Sampler seconds | Full / cached / predicted | Whole-image SSIM vs same-backend baseline |
| --- | ---: | --- | ---: |
| Sage baseline, no target skipping | 34.231 | 40 / 0 / 0 | 1 |
| Block, threshold 0.08 | 20.107 | 21 / 19 / 0 | 0.9010 |
| Block 0.08 + Spectrum 0.25 | 20.182 | 21 / 19 / 0 | — |
| Spectrum alone, guard 0.25 | 24.942 | 26 / 0 / 14 | 0.9891 |
| Conservative Block, threshold 0.03 | 30.029 | 34 / 6 / 0 | — |
| Conservative Block 0.03 + Spectrum 0.08 | 25.089 | 26 / 6 / 8 | 0.9893 |
| Conservative combination + forced Sol | 26.209 | 26 / 4 / 10 | — |

The conservative combination reduced sampler time by about **27%** versus baseline and **16%** versus conservative Block alone. Block 0.08 is faster but visibly changes leg pose, clothing and objects; it is not the recommended portrait-edit preset. The conservative combination and standalone Spectrum remained visually much closer to baseline in this case. SSIM measures image structure, not identity recognition or universal quality. One image/seed does not establish general reliability.

The forced-Sol row used enabled=true, min_tokens=4096, tau=1.0 and logged **430 actual Sol kernel calls**. It ran successfully with prefix KV and both target-cache methods, but was slower than the same combination without Sol. The published edit workflow therefore bypasses Sol. The earlier unresolved 2048 restart remains outside this validation.

Core compiler **enabled** follow-up, otherwise the same flags: Sage baseline **34.431 s**, conservative combination **24.742 s**, with 6 cache hits and 8 forecasts. Both logged zero graph breaks/rogues. The corresponding output pixels exactly matched the compiler-disabled runs. A further Block threshold=0 run performed all 40 forwards and produced **pixel-identical output to the native baseline**, confirming the no-skip path after patched runs.

With **Kitchen as the selected backend and Sage bypassed**, Core compiler enabled: baseline **33.564 s**, conservative combination **24.544 s** (about 27% less sampler time), same-backend SSIM **0.98927** and PSNR **38.35 dB**. This is independent Kitchen validation, not a configuration where Sage overrides Kitchen.

The fix uses ModelPatcher object patches on the first/last block calls and bypasses them for reference/text-only passes. Core can therefore populate and reuse its prefix KV cache while target blocks are skipped. Cache-hit output still uses Core's timestep embedding and output head. Each algorithm checks its own skip limit; adding Spectrum no longer reduces Block's limit from 2 to 1. Temporary anchors remain on the compute device until the residual is formed, avoiding an extra full anchor transfer to CPU. Core source files are unchanged.

Published canvas: [Qwen21_T8_1024_Edit.json](workflows/Qwen21_T8_1024_Edit.json). It uses Sage + Block 0.03 + Spectrum 0.08, with Kitchen and Sol bypassed. Select your own reference image; the test photograph is not shipped. Existing node IDs and parameters are unchanged. Restart ComfyUI after updating the plugin.

The final layout was exported, re-imported and run again: **24.829 s**, 26 full / 6 cached / 8 predicted, PNG saved successfully. Updated T2I canvases were also re-exported, re-imported and run on 0.1.3 with the same compiler-enabled server: Kitchen + Block **10.470 s** (11/25 cached), Kitchen + Spectrum **12.417 s** (9/25 predicted). No new T2I baseline was measured in this final check, so these rows verify execution rather than establish a new speed ratio. The Sol T2I canvas only received explanatory/layout updates and a new import check; its sampling settings retain the earlier completed 25-step validation.

Optional edit reproduction with an existing Playwright installation and idle dedicated server:

```text
node tests/canvas_edit.cjs --workflow workflows/Qwen21_T8_1024_Edit.json --mode combined --threshold 0.03 --guard 0.08 --backend sage
```

The test script fixes the local reference filename to `10A.jpg`, seed 42 and the tested portrait dimensions; it never downloads the photograph. Set up that local test input first. This script also requires the local 8189 service to be reachable and idle, to avoid overlapping the user's normal queue. The exported workflow itself has no such server/filename restriction: upload any reference through LoadImage.
