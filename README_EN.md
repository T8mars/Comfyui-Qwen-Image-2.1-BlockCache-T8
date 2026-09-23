# Comfyui-Qwen-Image-2.1-BlockCache-T8

[简体中文](README.md) | English

Four independent MODEL nodes for **native ComfyUI Qwen-Image-2.1**: Block Cache, Spectrum, Sage Attention and Sol Attention, experimental version `0.1.2`. Keep the official loader, text conditioning, sampler, and VAE. No Diffusers wrapper is introduced.

Real 7B INT8 T2I sampling, timing comparisons at 512/1024, and browser-canvas execution have been tested. Prompt/seed coverage is limited; quality and speed are not guaranteed for every workflow. **The workstation restarted during a 2048 full-model Sol test; the cause is unresolved. Sol is disabled by default and is not a validated acceleration recommendation.**

## Drag-and-drop canvas workflows

- [1024 T2I: Kitchen + Block Cache](workflows/Qwen21_T8_1024_T2I.json)
- [1024 T2I: Kitchen + Spectrum](workflows/Qwen21_T8_1024_Spectrum.json)

Download the raw JSON, drop it onto ComfyUI, select your local model filenames, and click Run. These are **frontend workflows with layouts and links, not API JSON**. Both were imported and executed by clicking Run in the real browser frontend, completing PNG output. Purple nodes are bypassed; select and press `Ctrl+B` to toggle. Leave Sol's separate `enabled=false` control off.

An additional [experimental 1024 Sol workflow](workflows/Qwen21_T8_1024_Sol.json) completed 25 steps from the real canvas. With Core compilation disabled and extra VRAM headroom, serial comparisons averaged 16.61 s for Kitchen versus 15.96 s for Sol (about 4% less sampler time), but introduced garbled small text and changed lid details. This file explicitly enables Sol (`enabled=true, min_tokens=4096`); it is not the recommended default. See [full conditions and limits](BENCHMARKS.md#serial-1024-sol-follow-up--1024-串行复测). This does not establish 2048 safety.

RTX 4060 Ti 16GB, 1024², 25 steps, same prompt/seed: native sampling **18–19 s**, Block **11.3 s**, Kitchen + Block **9.6 s**, repaired Spectrum **13.3 s**. These are sampler timings, not total workflow timings. See [conditions, output differences and limits](BENCHMARKS.md). Run tests strictly one at a time; 2048 stress testing is suspended.

## Installation and wiring

Search ComfyUI-Manager for **Qwen Image 2.1 BlockCache T8** or `qwen-image-21-blockcache-t8`, and verify publisher `t8star`. Check the [Comfy Registry page](https://registry.comfy.org/publishers/t8star/nodes/qwen-image-21-blockcache-t8) for publication status. If it is not indexed yet, refresh the list or install manually:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-Qwen-Image-2.1-BlockCache-T8.git
```

Restart the backend. Nodes appear under `T8/Qwen Image 2.1`. Requires native Qwen2.1 support (Core baseline 0.37.0) and the matching Comfy Kitchen. The Sage node requires an existing compatible SageAttention installation. Nothing downloads or installs automatically.

If you already installed `Comfyui-Qwen-Image-2.1-T8`, do not install a duplicate. Run these commands inside the existing plugin folder; keeping its old folder name is fine. All four node IDs remain unchanged:

```bash
git remote set-url origin https://github.com/T8mars/Comfyui-Qwen-Image-2.1-BlockCache-T8.git
git pull --ff-only
```

Start with the [official workflow](https://github.com/Comfy-Org/workflow_templates/blob/main/templates/image_qwen_image_2_1_t2i.json). Insert nodes on the MODEL connection, then connect the final MODEL to the native sampler/guider/scheduler. Prompts, references, RGBA handling, and VAE remain native.

```text
Native diffusion loader → optional static LoRA
  → optional KJ generic Sage / T8 Sage / native Model Attention Backend
  → optional T8 Sol Attention
  → optional T8 Block Cache
  → optional T8 Spectrum
  → native sampling
```

For Kitchen, use native `Model Attention Backend` with `comfy kitchen attention`. KJ means generic `Patch Sage Attention KJ`, not an H3-specific memory-efficient patch. Existing generic Sol nodes can remain upstream, but typically decline Qwen2.1's rectangular Q/K; this project's Qwen-specific adapter performs actual Sol dispatch.

## Nodes

| Node | Behavior | Experimental defaults |
| --- | --- | --- |
| Qwen Image 2.1 Block Cache (T8) | Recompute Block 0; reuse target tail residual to skip Blocks 1–31 when stable | threshold `0.08`, window `0.10–0.85`, consecutive limit `2` |
| Qwen Image 2.1 Spectrum (T8) | Chebyshev/ridge prediction of target tail residual from real full forwards; retain Block 0 stability guard | history `4`, degree `2`, ridge `0.01`, guard `0.25`, window `0.15–0.85`, consecutive limit `1` |
| Qwen Image 2.1 Sage Attention (T8) | Native masked/rectangular Sage adapter and fallback behavior | no extra inputs |
| Qwen Image 2.1 Sol Attention (T8) | Experimental sparse attention; limited 1024 test passed, 2048 restart unresolved; off by default | enabled `false`, tau `1.0`, min_tokens `12288`, window `0.15–0.85` |

Block Cache and T8 Spectrum work independently or together, in either order. Cache reuse has priority; prediction is attempted next. Only real full passes enter history. The stricter consecutive skip limit wins; CPU storage wins if either node selects CPU; the smaller memory budget wins.

Default retained-cache budget is `1024 MiB` on CPU. Older streams are evicted when the budget is exceeded. Temporary anchors, reconstructed outputs, and model allocations are additional memory. At 2048² and bf16, each target hidden residual is about 128 MiB; four spectral anchors are about 512 MiB per conditioning stream, potentially doubled by CFG. Forecasting accumulates FP32 chunks of 512 tokens instead of a feature-sized regression matrix.

Higher cache thresholds or Sol tau are more aggressive and may alter quality. Smaller metric stride checks more values at higher cost. Sampling windows use native `percent_to_sigma`.

## Qwen-specific behavior and limits

- Requires `QwenImage21Transformer2DModel`; old Qwen-Image, 2512, and H3 are different architectures.
- Target-only residual storage; full forwards retain native fused RMS/RoPE, AdaLN, SwiGLU, quantized operations, and offload.
- State exists only inside one sampling call. UUID/CFG, geometry, dtype/device, and reference layout separate streams. Missing identity/sigma prevents reuse; repeated/reversed sigma refreshes history. Completion, failure, and cancellation release state.
- **Core disables its native prefix KV cache when Block/Spectrum installs block hooks.** Reference-heavy editing may become slower. Sage/Sol alone do not install those hooks and retain the native prefix-cache path. Compare against a no-cache baseline.
- Static LoRAs stay native. Scheduled weight `hook_patches` disable residual reuse and forecasting for that run.
- Sol prepends dummy queries only, keeps K/V intact, discards dummy outputs, and forces prefix KV and the mixed 64-token query boundary exact. Masks, reference segments, short sequences, FP32, unavailable kernels, and disabled low precision retain the dense backend.
- Sol requires explicit `enabled=true`. A 1024² target has about 4096 tokens and also stays below the default sparse threshold. The 2048 run did not complete; no stability/speedup claim is made and enabling it is not recommended.
- Foreign Spectrum, EasyCache/LazyCache, and block-modifying plugins do not share this state protocol. Detected conflicts fail clearly. Use this package's **T8 Spectrum** when combining with its Block Cache. Arbitrary third-party monkeypatches are not claimed to be detectable.
- Same-seed output can change with these approximate methods. CFG batching, schedule, resolution, and quantization affect quality and hit rate.

Logs report `full=N cache=N spectrum=N peak cache=N MiB` and `Sol: kernel=N dense=N`. Zero kernel calls mean Sol was not used.

## Validation

```powershell
python -m unittest discover -s tests -v
ruff check .
python -m compileall -q __init__.py nodes.py cache.py runtime.py attention.py tests
# Optional small GPU probe; ComfyUI must be importable via cwd or PYTHONPATH:
python tests/gpu_smoke.py
```

2026-09-23 baseline: Core `e638023d`, Kitchen `0.2.35`, Torch `2.10.0+cu130`; 39 CPU/static tests passed 50 serial rounds (1,950 executions, shuffled order), plus four mocked browser-lifecycle cases. These are not 50 full-model runs. Tiny randomly initialized native models match full-forward reference results for T2I, editing, and batching. Synthetic tail tests confirm actual block skipping and native output-head execution.

Cross-review fixes: non-finite stored residuals and reconstructed cache/Spectrum outputs fall back to real computation; mixed-sigma batches do not use Sol. Test submissions with unknown remote status retain the lock, and browsers close on errors. API tests accept one validated mode per invocation; GPU probes share the same lock. GPU timings have not been rerun after these checks were added; earlier measurements are historical, not new measurements of this revision.

RTX 4060 Ti small-tensor probe: BF16/FP16 compiled-vs-eager Sol relative L2 approximately `0.01225/0.01234`; this does not establish full-model Sol stability. Real INT8 7B DynamicVRAM sampling and canvas results are in [BENCHMARKS.md](BENCHMARKS.md). Fixed the default quadratic forecast being rejected by an overly strict extrapolation bound: Spectrum now predicts 9/25 steps in the 1024 example; Block caches 11/25.

## License and release status

Original plugin code: [Apache-2.0](LICENSE). Sources and attribution: [NOTICE](NOTICE.md). Qwen model materials retain their own license; no weights are included. Spectrum is a reduced Qwen-specific adaptation of spectral forecasting, not the complete paper controller.

Source repository: [T8mars/Comfyui-Qwen-Image-2.1-BlockCache-T8](https://github.com/T8mars/Comfyui-Qwen-Image-2.1-BlockCache-T8). Registry ID: `qwen-image-21-blockcache-t8`; publisher: `t8star`. The [official ComfyUI publishing workflow](https://docs.comfy.org/registry/publishing) publishes through GitHub Actions when `pyproject.toml` changes. A successful GitHub push does not mean Registry processing is complete; check the [publish job](https://github.com/T8mars/Comfyui-Qwen-Image-2.1-BlockCache-T8/actions/workflows/publish_action.yml) and Registry status.

`0.1.2`: fixes low-precision cache/Spectrum overflow and Sol windows for mixed-sigma batches; strengthens serial-test locking and cleanup; adds the real-canvas-tested experimental 1024 Sol workflow. Research notes, machine logs, models and generated images are excluded from releases.
