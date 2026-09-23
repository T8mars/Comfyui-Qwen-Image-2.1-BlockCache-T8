# Comfyui-Qwen-Image-2.1-BlockCache-T8

[简体中文](README.md) | English

Four independent MODEL nodes for **native ComfyUI Qwen-Image-2.1**: Block Cache, Spectrum, Sage Attention and Sol Attention, experimental version `0.1.4`. Keep the official loader, text conditioning, sampler, and VAE. No Diffusers wrapper is introduced.

`0.1.4` adds native-progress early/late thresholds, optional Sage/Kitchen routing, and two tested hybrid edit canvases. Old workflows retain fixed thresholds and plain Sage by default. Hybrid has not beaten Kitchen alone in the measured case; additive speedups are not claimed.

`0.1.3` fixes the image-edit slowdown: Block/Spectrum now preserve Core's reference/text prefix KV cache, and adding Spectrum no longer lowers Block's consecutive-hit limit. A real 7B INT8, 1MP portrait edit completed 40 steps from the browser canvas with Core compilation enabled: **34.43 s Sage baseline → 24.74 s conservative combined**, about 28% less sampler time; **33.56 s Kitchen baseline → 24.54 s combined**. The edit example uses Block `0.03` and Spectrum `0.08`, instead of the more aggressive T2I defaults.

Coverage is still limited to a few inputs/seeds, and approximate skipping changes details. **A 2048 full-model Sol test restarted the workstation; the cause remains unresolved. Sol is off by default.** Forcing Sol on in the 1MP edit did not add speed. Enabling every node is not necessarily fastest. See [conditions and image differences](BENCHMARKS.md#image-edit-fix-013).

## Drag-and-drop canvas workflows

- [1024 T2I: Kitchen + Block Cache](workflows/Qwen21_T8_1024_T2I.json)
- [1024 T2I: Kitchen + Spectrum](workflows/Qwen21_T8_1024_Spectrum.json)
- [1MP image edit: Sage + conservative Block/Spectrum](workflows/Qwen21_T8_1024_Edit.json)
- [New: 1MP image edit: hybrid + Block/Spectrum](workflows/Qwen21_T8_1024_Hybrid_Edit.json)
- [New: 1MP image edit: hybrid only, no skipped blocks](workflows/Qwen21_T8_1024_Hybrid_NoCache_Edit.json)

Download the raw JSON, drop it onto ComfyUI, select your local model filenames, and click Run. These are **frontend workflows with layouts and links, not API JSON**. All listed configurations were imported and executed by clicking Run in the real browser frontend, completing PNG output. Purple nodes are bypassed; select and press `Ctrl+B` to toggle. Leave Sol's separate `enabled=false` control off.

Both new files are in the repository's `workflows` directory. Their sampled settings are unchanged; only notes, layout, output prefixes and package metadata were tidied. The hybrid combination uses `constant` thresholds; select `two_stage` manually to use progress-based stages. Select your own reference photo; it is not distributed.

An additional [experimental 1024 Sol workflow](workflows/Qwen21_T8_1024_Sol.json) completed 25 steps from the real canvas. With Core compilation disabled and extra VRAM headroom, serial comparisons averaged 16.61 s for Kitchen versus 15.96 s for Sol (about 4% less sampler time), but introduced garbled small text and changed lid details. This file explicitly enables Sol (`enabled=true, min_tokens=4096`); it is not the recommended default. See [full conditions and limits](BENCHMARKS.md#serial-1024-sol-follow-up--1024-串行复测). This does not establish 2048 safety.

Historical T2I benchmark (0.1.1, RTX 4060 Ti 16GB, 1024², 25 steps): native sampling **18–19 s**, Block **11.3 s**, Kitchen + Block **9.6 s**, Spectrum **13.3 s**. These are not new 0.1.3 measurements. All speed figures refer to the sampler, not the entire workflow. Run tests one at a time; 2048 stress testing is suspended.

For editing, connect the VAE to **both** `TextEncodeQwenImage21` and the decoder. Connecting only the reference IMAGE omits the VAE reference latents. To retain the reference canvas size, use the encoder's LATENT output for sampling; this example intentionally uses portrait dimensions and permits recomposition. Reference photos are not distributed; select your own image.

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

Block Cache and T8 Spectrum work independently or together, in either order. Cache reuse has priority; prediction is attempted next. Only real full passes enter history. Each method checks skips since the last full pass against its own limit; Spectrum no longer lowers Block's limit from 2 to 1. CPU storage wins if either node selects CPU; the smaller memory budget wins. Combined gains depend on actual hits: `spectrum=0` means no forecasting benefit.

Default retained-cache budget is `1024 MiB` on CPU. Older streams are evicted when the budget is exceeded. Temporary anchors, reconstructed outputs, and model allocations are additional memory. At 2048² and bf16, each target hidden residual is about 128 MiB; four spectral anchors are about 512 MiB per conditioning stream, potentially doubled by CFG. Forecasting accumulates FP32 chunks of 512 tokens instead of a feature-sized regression matrix.

Higher cache thresholds or Sol tau are more aggressive and may alter quality. Smaller metric stride checks more values at higher cost. Sampling windows use native `percent_to_sigma`.

## Qwen-specific behavior and limits

### Sage / Kitchen routing (0.1.4)

The existing Sage node adds optional `backend_mode=sage_kitchen`: unmasked low-precision CUDA attention with at least 1024 query tokens and head dimension 128 uses Core's Kitchen adapter; other calls keep the native Sage path. Unavailable Kitchen falls back to Sage. The default `sage` preserves old workflows. Bypass the separate official Kitchen selector when using this mode; Block/Spectrum can still follow it.

This routes distinct calls rather than running the same attention twice, and does not relax cache thresholds. Sage's adapter may fall back to PyTorch for unsupported masks, so the `dense routes` log counts adapter routes, not guaranteed Sage kernel invocations. Core's existing Kitchen RMS/RoPE, AdaLN and SwiGLU fusion remains in use.

This is experimental, not a guarantee of beating Kitchen alone. Switching quantized attention backends can change the image; identical pixels and additive speedups are not promised.

Serial real-canvas 1MP edit, 40 steps: Sage `35.018/34.742 s`, hybrid `34.280/34.286 s`, Kitchen alone `34.269 s`. Hybrid was about 1.7% faster than the mean of those Sage runs but did not beat Kitchen. A hybrid + Block/Spectrum canvas run also completed. See [conditions and limitations](BENCHMARKS.md#attention-routing-014).

### Two-stage thresholds (0.1.4)

Both Block Cache and Spectrum support `threshold_mode=two_stage`. **All boundaries use native sampling progress converted through `percent_to_sigma`, never step or model-call counters.** The existing threshold becomes the early threshold. `late_threshold` controls the late stage; `split_ratio` is the early share of the active progress window.

```text
switch_progress = start_percent + (end_percent - start_percent) * split_ratio
```

With both nodes set to start=`0.15`, end=`0.85`, split=`0.5`: before 0.15 compute fully; `[0.15, 0.50)` uses each node's original threshold; `[0.50, 0.85)` uses its late threshold; from 0.85 compute fully. A split ratio of `0.3` switches at `0.36`, not 30% of the whole run. History requirements, change guards, and consecutive limits still apply; stages do not guarantee a number of hits.

- Block uses `residual_diff_threshold` early and `late_threshold` late; Spectrum uses `guard_threshold` early and `late_threshold` late.
- Each node has independent boundaries and thresholds. Align both active windows if the whole combination must compute the same initial/final regions fully. Setting one late threshold to zero disables only that node's late skipping; the other may still hit.
- Split `0` uses the late threshold throughout the active window; `1` uses the early threshold throughout. Default `constant` ignores staged controls and preserves old behavior. New optional inputs are appended; restart and reload an existing workflow to see them.

The staged feature has CPU tiny-model and real frontend import/save/prompt-parameter checks. Published 0.1.3 full-model speed figures below are for constant thresholds, not a new benchmark of staged presets.

### Model boundaries

- Requires `QwenImage21Transformer2DModel`; old Qwen-Image, 2512, and H3 are different architectures.
- Target-only residual storage; full forwards retain native fused RMS/RoPE, AdaLN, SwiGLU, quantized operations, and offload.
- State exists only inside one sampling call. UUID/CFG, geometry, dtype/device, and reference layout separate streams. Missing identity/sigma prevents reuse; repeated/reversed sigma refreshes history. Completion, failure, and cancellation release state.
- **ModelPatcher wraps the first/last blocks while preserving native prefix KV caching.** Reference/text-only passes never enter target skipping. Core source files remain unchanged; ModelPatcher restores the patched methods. Native `Qwen Image 2.1 Cache` device/precision settings remain owned by Core.
- Static LoRAs stay native. Scheduled weight `hook_patches` disable residual reuse and forecasting for that run.
- Sol prepends dummy queries only, keeps K/V intact, discards dummy outputs, and forces prefix KV and the mixed 64-token query boundary exact. Masks, reference segments, short sequences, FP32, unavailable kernels, and disabled low precision retain the dense backend.
- Sol requires explicit `enabled=true`. A 1024² target has about 4096 tokens and also stays below the default sparse threshold. The 2048 run did not complete; no stability/speedup claim is made and enabling it is not recommended.
- Foreign Spectrum, EasyCache/LazyCache, and block-modifying plugins do not share this state protocol. Detected conflicts fail clearly. Use this package's **T8 Spectrum** when combining with its Block Cache. Arbitrary third-party monkeypatches are not claimed to be detectable.
- Same-seed output can change with these approximate methods. CFG batching, schedule, resolution, and quantization affect quality and hit rate.

Logs report `full=N cache=N spectrum=N peak cache=N MiB` and `Sol: kernel=N dense=N`. Zero kernel calls mean Sol was not used.

## Validation

0.1.4: 59 CPU/static tests and four browser-cleanup mocks passed, covering progress stages, attention routing and the two published hybrid canvases. Six real edit canvas runs completed strictly serially. The new workflows retain sampled configurations; no new speed claim is made for staged thresholds.

```powershell
python -m unittest discover -s tests -v
ruff check .
python -m compileall -q __init__.py nodes.py cache.py runtime.py attention.py tests
# Optional small GPU probe; ComfyUI must be importable via cwd or PYTHONPATH:
python tests/gpu_smoke.py
```

0.1.3: 43 CPU/static tests passed, including actual Core prefix-KV reuse with skipped blocks, method restoration, no-skip batch=2 exact agreement, combined hit limits, and the edit canvas's VAE connections. Real 7B INT8 edits were imported and executed by clicking Run in the browser canvas, covering Block, Spectrum, combined, and a combination that actually invokes Sol. See [BENCHMARKS.md](BENCHMARKS.md).

Historical 0.1.2 review: Core `e638023d`, Kitchen `0.2.35`, Torch `2.10.0+cu130`; 39 CPU/static tests passed 50 serial rounds (1,950 executions), plus four mocked browser-lifecycle cases. These were not 50 full-model runs.

Non-finite residual/reconstruction fallback, mixed-sigma Sol guards, cleanup, and serial-test locking remain in place. The new edit measurements include these checks; earlier T2I measurements retain their historical provenance.

RTX 4060 Ti small-tensor probe: BF16/FP16 compiled-vs-eager Sol relative L2 approximately `0.01225/0.01234`; this does not establish full-model Sol stability. Real INT8 7B DynamicVRAM sampling and canvas results are in [BENCHMARKS.md](BENCHMARKS.md). Fixed the default quadratic forecast being rejected by an overly strict extrapolation bound: Spectrum now predicts 9/25 steps in the 1024 example; Block caches 11/25.

## License and release status

Original plugin code: [Apache-2.0](LICENSE). Sources and attribution: [NOTICE](NOTICE.md). Qwen model materials retain their own license; no weights are included. Spectrum is a reduced Qwen-specific adaptation of spectral forecasting, not the complete paper controller.

Source repository: [T8mars/Comfyui-Qwen-Image-2.1-BlockCache-T8](https://github.com/T8mars/Comfyui-Qwen-Image-2.1-BlockCache-T8). Registry ID: `qwen-image-21-blockcache-t8`; publisher: `t8star`. The [official ComfyUI publishing workflow](https://docs.comfy.org/registry/publishing) publishes through GitHub Actions when `pyproject.toml` changes. A successful GitHub push does not mean Registry processing is complete; check the [publish job](https://github.com/T8mars/Comfyui-Qwen-Image-2.1-BlockCache-T8/actions/workflows/publish_action.yml) and Registry status.

`0.1.3`: fixes the edit slowdown caused by losing Core prefix KV caching; fixes combined hit limits; reduces residual CPU transfers; adds a tested edit canvas. Research notes, machine logs, models, reference photos and generated images are excluded from releases.
