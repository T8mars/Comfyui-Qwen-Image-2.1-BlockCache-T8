# Comfyui-Qwen-Image-2.1-BlockCache-T8

[简体中文](README.md) | English

Experimental acceleration nodes for **native ComfyUI Qwen-Image-2.1**. Keep the official model loader, conditioning, sampler, and VAE; add attention backends or block skipping on the MODEL connection.

## Install

Requires a Qwen-Image-2.1-capable ComfyUI (minimum `0.37.0`). Search the node manager for `qwen-image-21-blockcache-t8` and verify publisher `t8star`, or install manually:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/T8mars/Comfyui-Qwen-Image-2.1-BlockCache-T8.git
```

Restart ComfyUI. T8 Sage requires a compatible SageAttention in the same Python environment. Kitchen uses ComfyUI's own `Model Attention Backend` node. If you installed the former `Comfyui-Qwen-Image-2.1-T8` repository, update that installation's Git remote instead of installing a second copy.

## Drag-and-drop workflows

These are **frontend canvas JSONs**, not API prompts. After importing, select your local models; for editing, provide your own reference image.

| Use | Workflow |
| --- | --- |
| Text to image | [Kitchen + Block Cache](workflows/Qwen21_T8_1024_T2I.json) · [Kitchen + Spectrum](workflows/Qwen21_T8_1024_Spectrum.json) |
| Image edit | [Sage + Block/Spectrum](workflows/Qwen21_T8_1024_Edit.json) · [Hybrid + Block/Spectrum](workflows/Qwen21_T8_1024_Hybrid_Edit.json) · [Hybrid only, no skipped blocks](workflows/Qwen21_T8_1024_Hybrid_NoCache_Edit.json) |
| Experimental | [Sol text-to-image](workflows/Qwen21_T8_1024_Sol.json) |

For editing, connect both the reference image and the VAE to `TextEncodeQwenImage21`; also connect the VAE to the decoder. Omitting the encoder-side VAE can distort the edit. Use the encoder's LATENT output if you need the source canvas dimensions. Select a purple bypassed node and press `Ctrl+B` to enable it; Sol also has a separate `enabled` switch.

## Wiring and settings

```text
Official model loader → optional LoRA → one attention backend
  → optional T8 Block Cache → optional T8 Spectrum → official sampler
```

Choose the official Kitchen backend, generic KJ Sage, or T8 Sage. Their speedups do **not** add together. T8 Sage's experimental `backend_mode=sage_kitchen` routes by attention shape; do not also chain a Kitchen selector. In our test it did not beat Kitchen alone. T8 Sol is off by default and is not recommended for routine use.

| Node | What it does | Starting point |
| --- | --- | --- |
| Block Cache | Checks the first block; reuses later-block residuals when stable | For edits, try `residual_diff_threshold=0.03` |
| Spectrum | Predicts residuals from full-forward history; composes with this package's Block Cache | For edits, try `guard_threshold=0.08` |
| Sage Attention | Native Sage adapter with optional Sage/Kitchen routing | Default `backend_mode=sage` |
| Sol Attention | Experimental sparse attention | Default `enabled=false` |

Higher Block/Spectrum thresholds skip more readily but can change identity, text, and details. `start_percent` and `end_percent` bound their active window; outside it, computation is full. `max_consecutive_hits` limits consecutive skips. Compare against an unaccelerated result before combining nodes. These caches support native Qwen-Image-2.1 only and do not share state with third-party EasyCache/Spectrum.

### Early/late thresholds

Set Block or Spectrum `threshold_mode=two_stage`: the existing threshold controls the early stage, `late_threshold` the late stage, and `split_ratio` the early share **within the active window**:

```text
switch_progress = start_percent + (end_percent - start_percent) × split_ratio
```

For start `0.15`, end `0.85`, and split `0.5`, the threshold changes at sampling progress `0.50`, **not at a step count**. Default `constant` preserves old workflows. Two-stage behavior has functional tests, but the speed figures below use constant thresholds.

## Measured result and limits

For one 7B INT8, 1MP portrait edit at 40 steps with Core compilation: Sage baseline **34.43 → 24.74 s** and Kitchen baseline **33.56 → 24.54 s** with Block `0.03` + Spectrum `0.08`. These are **sampler times** for limited images/seeds, not whole-workflow times or quality guarantees. Hybrid Sage/Kitchen was roughly equal to Kitchen alone in this case. A 2048 Sol test restarted the workstation; the cause remains unknown. See [BENCHMARKS.md](BENCHMARKS.md) for conditions, other results, and image differences.

Code is [Apache-2.0](LICENSE). Model weights are not distributed; see [NOTICE](NOTICE.md) for attribution.
