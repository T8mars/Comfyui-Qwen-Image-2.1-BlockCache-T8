# 真实模型与画布验证 / Real-model and canvas validation

2026-09-23，RTX 4060 Ti 16 GB / Windows，ComfyUI `e638023d` (0.37.0)，Frontend 1.53.6，Torch 2.10.0+cu130，Kitchen 0.2.35，Aimdo 0.5.5。使用原生 DynamicVRAM，`--reserve-vram 3 --vram-headroom 1 --preview-method none`。不修改 Core，不下载模型。

Weights: `qwen_image_2.1_int8_convrot.safetensors` (7B), `qwen3vl_8b_fp8_scaled.safetensors`, `qwen_image_2.1_vae_bf16.safetensors`. Seed 42, CFG 1, Euler/simple, 25 steps, batch 1. Prompt is the red-teapot example in the canvas workflows. Cache storage is CPU. This is limited T2I evidence, not a guarantee across prompts, hardware, editing or model formats.

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

Example: Kitchen + Block reduces sampling time by about 47–50% versus this native default, or about 39% versus Kitchen alone. **This is not a promise of 2× total workflow speed.** Prefer Block alone when preserving the existing attention backend; replacing attention can itself change the image substantially. Enabling every node is not necessarily faster: combined caching uses the stricter shared consecutive-skip cap.

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

Both files in `workflows/` are **frontend workflow JSON**, containing nodes, links, positions and widget values, not API prompt dictionaries. They were created/serialized by the real ComfyUI frontend, imported through its file input, and executed by clicking the visible Run button in Chromium. Both completed sampling, VAE decoding and PNG saving. Returned PNG metadata contains the canvas workflow as well. No missing-node dialog or submission error occurred.

- `Qwen21_T8_1024_T2I.json`: Kitchen + Block active; Sage/Sol/Spectrum bypassed.
- `Qwen21_T8_1024_Spectrum.json`: Kitchen + Spectrum active; Block/Sage/Sol bypassed.

The saved layout/notes were subsequently tidied without changing either tested active sampling path. Sol's additional `enabled=false` control is inactive in both workflows. Models must already exist locally; choose your filenames after import. Sage is optional for these two workflows.

## Sol limitation and test safety / Sol 限制与测试安全

Small BF16/FP16 compiled Sol kernel tests passed, but **full-model Sol speed and stability are unverified**. A 2048 Kitchen baseline finished in 93.41 s; the following Kitchen + Sol run stopped at step 14/25 when the workstation restarted. No Python traceback establishes the cause. Do not infer a Sol speedup from partial step timings or claim the restart was definitively caused by a particular component.

Sol now defaults to `enabled=false`, and both example workflows bypass it. Large-image testing is suspended. No 2048 workflow is distributed. **All future tests must be strictly serial**, one runner/workflow at a time. The API and canvas test scripts share an exclusive lock and refuse to enqueue when ComfyUI is busy. A stale lock after a crash must be reviewed and removed only after verifying no test is running. These guards do not impose a hard GPU-memory limit or guarantee driver stability.

## Reproduction

Use a dedicated idle localhost ComfyUI server. Start with 512, one mode per invocation:

```text
python tests/benchmark_native.py --size 512 --steps 25 --modes baseline
python tests/benchmark_native.py --size 512 --steps 25 --modes block
```

For frontend verification, import a file from `workflows/` and click Run. `tests/canvas_smoke.cjs --run` automates that with an existing Playwright installation; it is optional and not a runtime dependency. Do not run it while another benchmark is active. `tests/compare_outputs.py` optionally compares local PNGs with Pillow/NumPy/scikit-image. No models, test outputs, local research notes or machine logs are included in the repository.
