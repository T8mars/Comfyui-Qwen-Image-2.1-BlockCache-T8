from dataclasses import dataclass
import logging

import torch
import comfy.model_management
import comfy.model_patcher
import comfy.model_prefetch as prefetch
import comfy.patcher_extension as patches

from .attention import make_override
from .cache import CacheHit, CacheRuntime, ForwardState


KEY = "qwen_image21_t8"
FRAME = KEY + "_forward"
RUNTIME = KEY + "_runtime"


@dataclass
class SamplingRuntime:
    cache: CacheRuntime | None
    sol_window: tuple | None
    stats: dict


def validate_patches(options, block_count):
    config = options.get(KEY, {})
    if not ("block" in config or "spectrum" in config):
        return
    if "easycache" in options:
        raise ValueError("Qwen Image 2.1 T8 cache cannot share a model with EasyCache/LazyCache")
    replacements = options.get("patches_replace", {}).get("dit", {})
    for index in range(block_count):
        patch = replacements.get(("single_block", index))
        if patch is not None and not isinstance(patch, BoundaryPatch):
            raise ValueError("Qwen Image 2.1 T8 cache conflicts with another single_block replacement")
    for hook in ("single_block", "post_input", "attn1_patch"):
        if options.get("patches", {}).get(hook):
            raise ValueError(f"Qwen Image 2.1 T8 cache cannot skip external {hook} hooks")
    if any(w is not diffusion_wrapper for w in patches.get_all_wrappers(patches.WrappersMP.DIFFUSION_MODEL, options)):
        raise ValueError("Qwen Image 2.1 T8 cache conflicts with an external diffusion wrapper; use T8 Spectrum")


def validate_model(model):
    config = model.model_options["transformer_options"].get(KEY, {})
    if "block" not in config and "spectrum" not in config:
        return
    if model.model_options.get("model_function_wrapper") is not None:
        raise ValueError("Qwen Image 2.1 T8 cache conflicts with an external model wrapper; use T8 Spectrum for combined caching")
    if any(w is not diffusion_wrapper for w in model.get_all_wrappers(patches.WrappersMP.DIFFUSION_MODEL)):
        raise ValueError("Qwen Image 2.1 T8 cache requires exclusive diffusion-wrapper ownership; use T8 Spectrum")


def sample_wrapper(executor, *args, **kwargs):
    guider = executor.class_obj
    original = guider.model_options
    local = comfy.model_patcher.create_model_options_clone(original)
    options = local["transformer_options"]
    config = options[KEY]
    model = guider.model_patcher.model
    validate_model(guider.model_patcher)
    validate_patches(options, len(model.diffusion_model.transformer_blocks))
    cache = CacheRuntime(config.get("block"), config.get("spectrum"), model.model_sampling) if "block" in config or "spectrum" in config else None
    if cache is not None and guider.model_patcher.hook_patches:
        logging.info("Qwen Image 2.1 T8: scheduled weight hooks detected; computing all blocks")
        cache = None
    sol = config.get("sol")
    window = (float(model.model_sampling.percent_to_sigma(sol.start_percent)), float(model.model_sampling.percent_to_sigma(sol.end_percent))) if sol else None
    runtime = SamplingRuntime(cache, window, {})
    options[RUNTIME] = runtime
    guider.model_options = local
    try:
        return executor(*args, **kwargs)
    finally:
        if cache is not None:
            logging.info("Qwen Image 2.1 T8: full=%d cache=%d spectrum=%d peak cache=%.1f MiB", cache.full, cache.hits, cache.forecasts, cache.peak_bytes / 1048576)
            cache.clear()
        if sol:
            logging.info("Qwen Image 2.1 T8 Sol: kernel=%d dense=%d", runtime.stats.get("sol", 0), runtime.stats.get("sol_dense", 0))
        guider.model_options = original


def finish_prefetch(model, device):
    prefetch.prefetch_queue_pop(None, device, None, malloc_scope="block")
    block_ids = {id(b) for b in model.transformer_blocks}
    for queue in reversed(prefetch.PREFETCH_QUEUES):
        if not any(id(entry[1][0] if isinstance(entry, tuple) else entry) in block_ids for entry in queue):
            continue
        for entry in queue:
            if isinstance(entry, tuple):
                stream, (module, modules) = entry
                if stream is not None:
                    stream.wait_stream(comfy.model_management.current_stream(device))
                if modules is not None:
                    prefetch.cleanup_prefetched_modules(module, modules)
        queue[:] = [None]
        break
    prefetch.malloc_graph_end()


def diffusion_wrapper(executor, x, timestep, context, ref_latents=None, image_slots=None, transformer_options=None, **kwargs):
    options = dict(transformer_options or {})
    config = options.get(KEY, {})
    runtime = options.get(RUNTIME)
    target_tokens = x.shape[-2] * x.shape[-1]
    refs = list(ref_latents or [])
    total_tokens = target_tokens + context.shape[1] + sum(r.shape[-2] * r.shape[-1] for r in refs)
    sigmas = options.get("sigmas")
    sigma = float(sigmas.flatten()[0]) if isinstance(sigmas, torch.Tensor) and sigmas.numel() else None
    uniform_sigma = sigma is not None and bool((sigmas == sigma).all())

    progress = None
    if runtime and runtime.sol_window and sigma is not None:
        start, end = runtime.sol_window
        if end <= sigma <= start:
            progress = config["sol"].start_percent
    if config.get("sage") or config.get("sol"):
        options["optimized_attention_override"] = make_override(
            executor.class_obj, options.get("optimized_attention_override"), config.get("sage", False),
            config.get("sol"), target_tokens, total_tokens, progress, runtime.stats if runtime else {},
        )
    cache = runtime.cache if runtime else None
    options.pop(FRAME, None)
    if cache is not None:
        validate_patches(options, len(executor.class_obj.transformer_blocks))
        uuids = options.get("uuids")
        active = bool(uuids) and all(value is not None for value in uuids) and uniform_sigma
        frame = None
        if active:
            key = (tuple(map(str, uuids)), tuple(options.get("cond_or_uncond", [])), tuple(x.shape), x.dtype, x.device,
                   tuple(context.shape), tuple(tuple(r.shape) for r in refs), tuple(image_slots or []))
            stream = cache.stream(key, sigma)
            active = any(sigma > end for _, end in cache.windows.values())
            if not active:
                stream.clear()
            frame = ForwardState(cache, stream, sigma, target_tokens, active)
        options[FRAME] = frame
        cache.full += 1
    try:
        return executor(x, timestep, context, ref_latents, image_slots, options, **kwargs)
    except CacheHit as hit:
        model = executor.class_obj
        finish_prefetch(model, x.device)
        hidden = model.norm_out(hit.target, hit.temb[:-1])
        output = model.proj_out(hidden)
        return output.transpose(1, 2).reshape(x.shape[0], model.out_channels, x.shape[-2], x.shape[-1])


class BoundaryPatch:
    def __init__(self, first):
        self.first = first

    def __call__(self, args, extra):
        frame = args["transformer_options"].get(FRAME)
        if frame is None or not frame.active:
            return extra["original_block"](args)
        cache = frame.runtime
        if self.first:
            with prefetch.pause_malloc_graph():
                before = cache.sample(args["img"][:, -frame.target_tokens:])
            out = extra["original_block"](args)
            with prefetch.pause_malloc_graph():
                target = out["img"][:, -frame.target_tokens:]
                frame.indicator = cache.sample(target) - before
                replay, _ = cache.replay(frame.stream, frame.indicator, frame.sigma, target)
                if replay is not None:
                    cache.full -= 1
                    raise CacheHit(replay, args["vec"])
                if target.numel() * target.element_size() > cache.budget:
                    frame.active = False
                    frame.stream.clear()
                else:
                    frame.anchor = cache.copy(target)
            return out
        out = extra["original_block"](args)
        with prefetch.pause_malloc_graph():
            cache.store(frame.stream, frame.indicator, frame.sigma, out["img"][:, -frame.target_tokens:], frame.anchor)
            frame.anchor = None
        return out


def install(model, name, value):
    cloned = model.clone()
    options = cloned.model_options["transformer_options"].copy()
    config = dict(options.get(KEY, {}))
    config[name] = value
    options[KEY] = config
    validate_patches(options, len(model.model.diffusion_model.transformer_blocks))
    cloned.model_options["transformer_options"] = options
    if "block" in config or "spectrum" in config:
        cloned.set_model_patch_replace(BoundaryPatch(True), "dit", "single_block", 0)
        cloned.set_model_patch_replace(BoundaryPatch(False), "dit", "single_block", len(model.model.diffusion_model.transformer_blocks) - 1)
    for kind, wrapper in ((patches.WrappersMP.OUTER_SAMPLE, sample_wrapper), (patches.WrappersMP.DIFFUSION_MODEL, diffusion_wrapper)):
        cloned.remove_wrappers_with_key(kind, KEY)
        cloned.add_wrapper_with_key(kind, KEY, wrapper)
    validate_model(cloned)
    return cloned
