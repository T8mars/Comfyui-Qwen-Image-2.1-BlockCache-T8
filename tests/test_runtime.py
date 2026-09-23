import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
import comfy.options

comfy.options.enable_args_parsing(False)
import comfy.cli_args

comfy.cli_args.args.cpu = True
import comfy.model_patcher
import comfy.ops
import comfy.patcher_extension as pe
from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel

if "qwen21_t8" not in sys.modules:
    spec = importlib.util.spec_from_file_location("qwen21_t8", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from qwen21_t8 import cache, nodes, runtime

torch.set_num_threads(2)


class Sampling:
    def percent_to_sigma(self, percent):
        return 1 - percent


class AddBlock(torch.nn.Module):
    def __init__(self, amount):
        super().__init__()
        self.amount = amount
        self.calls = 0

    def forward(self, x, mod, pe, attn_fn, prefix_len, transformer_options):
        self.calls += 1
        return x.add_(self.amount)


def tiny_model(additive=False):
    torch.manual_seed(13)
    model = QwenImage21Transformer2DModel(in_channels=4, out_channels=4, num_layers=3, num_attention_heads=1,
                                         attention_head_dim=128, context_in_dim=8, mlp_ratio=1,
                                         dtype=torch.float32, device="cpu", operations=comfy.ops.manual_cast)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.normal_(0, 0.03)
    if additive:
        model.transformer_blocks = torch.nn.ModuleList([AddBlock(0.1), AddBlock(0.2), AddBlock(0.3)])
    holder = torch.nn.Module()
    holder.diffusion_model = model
    holder.model_sampling = Sampling()
    return comfy.model_patcher.ModelPatcher(holder, torch.device("cpu"), torch.device("cpu"))


def run_sampling(patcher, sigmas, uuids=None, edit=False, batch=1, error=False):
    original_options = comfy.model_patcher.create_model_options_clone(patcher.model_options)
    original_options["transformer_options"]["wrappers"] = patcher.wrappers
    guider = types.SimpleNamespace(model_patcher=patcher, model_options=original_options)
    model = patcher.model.diffusion_model
    torch.manual_seed(24)
    x = torch.randn(batch, 4, 2, 3)
    context = torch.randn(batch, 4, 8)
    refs = [torch.randn(batch, 4, 2, 2)] if edit else None
    outputs, runtimes = [], []

    def execute():
        run = guider.model_options["transformer_options"][runtime.RUNTIME]
        runtimes.append(run)
        for i, sigma in enumerate(sigmas):
            options = dict(guider.model_options["transformer_options"])
            options.update(sigmas=torch.full((batch,), sigma), uuids=[uuids[i] if uuids else "positive"], cond_or_uncond=[0])
            if error:
                raise RuntimeError("test cancellation")
            outputs.append(model(x.clone(), torch.full((batch,), sigma), context, refs, [2] if edit else None, options))
        return outputs

    executor = pe.WrapperExecutor.new_class_executor(execute, guider, [runtime.sample_wrapper])
    patcher.patch_model(load_weights=False)
    try:
        if error:
            with unittest.TestCase().assertRaisesRegex(RuntimeError, "test cancellation"):
                executor.execute()
        else:
            executor.execute()
    finally:
        patcher.unpatch_model(unpatch_weights=False)
    assert guider.model_options is original_options
    return outputs, runtimes[0]


class NativeTests(unittest.TestCase):
    def test_canvas_examples_have_valid_links_and_safe_defaults(self):
        for name, active in (("T2I", "QwenImage21BlockCacheT8"), ("Spectrum", "QwenImage21SpectrumT8"), ("Sol", "QwenImage21SolAttentionT8")):
            workflow = json.loads((ROOT / "workflows" / f"Qwen21_T8_1024_{name}.json").read_text(encoding="utf-8"))
            by_id = {node["id"]: node for node in workflow["nodes"]}
            by_type = {node["type"]: node for node in workflow["nodes"]}
            for link, source, source_slot, dest, dest_slot, kind in workflow["links"]:
                self.assertIn(link, by_id[source]["outputs"][source_slot]["links"])
                self.assertEqual(by_id[dest]["inputs"][dest_slot]["link"], link)
            self.assertEqual(by_type[active]["mode"], 0)
            sol = by_type["QwenImage21SolAttentionT8"]
            self.assertEqual(sol["mode"], 0 if name == "Sol" else 4)
            self.assertEqual(sol["widgets_values_named"]["enabled"], name == "Sol")
            if name == "Sol":
                self.assertEqual(sol["widgets_values_named"]["min_tokens"], 4096)
                for bypassed in ("QwenImage21BlockCacheT8", "QwenImage21SpectrumT8", "QwenImage21SageAttentionT8"):
                    self.assertEqual(by_type[bypassed]["mode"], 4)
            self.assertEqual(by_type["TextEncodeQwenImage21"]["widgets_values_named"]["resolution"], 1024)
            sampler = by_type["KSampler"]["widgets_values_named"]
            self.assertEqual((sampler["seed"], sampler["steps"], sampler["cfg"]), (42, 25, 1))

    def test_edit_canvas_has_reference_vae_and_conservative_cache(self):
        workflow = json.loads((ROOT / "workflows" / "Qwen21_T8_1024_Edit.json").read_text(encoding="utf-8"))
        by_id = {node["id"]: node for node in workflow["nodes"]}
        by_type = {node["type"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}
        for link, source, source_slot, dest, dest_slot, kind in workflow["links"]:
            self.assertIn(link, by_id[source]["outputs"][source_slot]["links"])
            self.assertEqual(by_id[dest]["inputs"][dest_slot]["link"], link)
        for kind, name, source in (("TextEncodeQwenImage21", "vae", "VAELoader"),
                                   ("TextEncodeQwenImage21", "images.image_1", "LoadImage"),
                                   ("VAEDecode", "vae", "VAELoader")):
            socket = next(value for value in by_type[kind]["inputs"] if value["name"] == name)
            self.assertEqual(by_id[links[socket["link"]][1]]["type"], source)
        for kind in ("QwenImage21BlockCacheT8", "QwenImage21SpectrumT8", "QwenImage21SageAttentionT8"):
            self.assertEqual(by_type[kind]["mode"], 0)
        self.assertEqual(by_type["QwenImage21BlockCacheT8"]["widgets_values_named"]["residual_diff_threshold"], 0.03)
        self.assertEqual(by_type["QwenImage21SpectrumT8"]["widgets_values_named"]["guard_threshold"], 0.08)
        self.assertEqual(by_type["QwenImage21SolAttentionT8"]["mode"], 4)
        self.assertFalse(by_type["QwenImage21SolAttentionT8"]["widgets_values_named"]["enabled"])
        self.assertEqual(by_type["ModelAttentionBackend"]["mode"], 4)

    def test_native_full_path_matches_reference_for_t2i_edit_and_batch(self):
        for edit, batch in ((False, 1), (True, 1), (True, 2)):
            with self.subTest(edit=edit, batch=batch):
                base = tiny_model()
                patched = nodes.QwenImage21BlockCacheT8.execute(base, residual_diff_threshold=0, start_percent=0, end_percent=1)[0]
                got, state = run_sampling(patched, [0.7, 0.6], edit=edit, batch=batch)
                torch.manual_seed(24)
                x, context = torch.randn(batch, 4, 2, 3), torch.randn(batch, 4, 8)
                refs = [torch.randn(batch, 4, 2, 2)] if edit else None
                for i, sigma in enumerate([0.7, 0.6]):
                    expected = base.model.diffusion_model(x.clone(), torch.full((batch,), sigma), context, refs, [2] if edit else None, {})
                    torch.testing.assert_close(got[i], expected, rtol=0, atol=0)
                self.assertEqual(state.cache.hits, 0)
                self.assertNotIn(runtime.KEY, base.model_options["transformer_options"])

    def test_cache_hit_really_skips_tail_and_preserves_native_head(self):
        model = tiny_model(additive=True)
        patched = nodes.QwenImage21BlockCacheT8.execute(model, start_percent=0, end_percent=1)[0]
        outputs, state = run_sampling(patched, [0.7, 0.6, 0.5, 0.4], edit=True)
        self.assertEqual([b.calls for b in model.model.diffusion_model.transformer_blocks], [4, 2, 2])
        self.assertEqual(state.cache.hits, 2)
        self.assertEqual(tuple(outputs[-1].shape), (1, 4, 2, 3))
        self.assertEqual(state.cache.streams, {})
        torch.manual_seed(24)
        x, context = torch.randn(1, 4, 2, 3), torch.randn(1, 4, 8)
        refs = [torch.randn(1, 4, 2, 2)]
        for output, sigma in zip(outputs, [0.7, 0.6, 0.5, 0.4]):
            expected = model.model.diffusion_model(x.clone(), torch.tensor([sigma]), context, refs, [2], {})
            torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)

    def test_cfg_uuid_isolation(self):
        patched = nodes.QwenImage21BlockCacheT8.execute(tiny_model(True), start_percent=0, end_percent=1)[0]
        _, state = run_sampling(patched, [0.7, 0.7, 0.6, 0.6], uuids=["positive", "negative", "positive", "negative"])
        self.assertEqual(state.cache.full, 2)
        self.assertEqual(state.cache.hits, 2)

    def test_block_cache_keeps_native_reference_kv_and_restores_forward(self):
        base = tiny_model()
        model = base.model.diffusion_model
        original = model.transformer_blocks[0].forward
        patched = nodes.QwenImage21BlockCacheT8.execute(base, residual_diff_threshold=1,
                                                       start_percent=0, end_percent=1)[0]
        self.assertNotIn("patches_replace", patched.model_options["transformer_options"])
        model.current_patcher = patched
        model.reset_prefix_cache(True)
        calls = []
        handles = [block.register_forward_pre_hook(
            lambda _, args, i=i: calls.append((i, args[0].shape[1], args[4])))
            for i, block in enumerate(model.transformer_blocks)]
        try:
            outputs, state = run_sampling(patched, [0.7, 0.6, 0.5], edit=True)
            self.assertEqual(state.cache.hits, 2)
            self.assertTrue(model.prefix_cache.filled(3))
            self.assertEqual(sum(n == prefix for _, n, prefix in calls), 3)
            self.assertEqual(sum(i == 0 and prefix == 0 for i, _, prefix in calls), 3)
            self.assertEqual(sum(i == 1 and prefix == 0 for i, _, prefix in calls), 1)
            self.assertEqual(outputs[-1].shape, (1, 4, 2, 3))
            self.assertEqual(model.transformer_blocks[0].forward, original)
        finally:
            for handle in handles:
                handle.remove()
            model.reset_prefix_cache(False)

    def test_spectrum_does_not_reduce_block_consecutive_limit(self):
        patched = nodes.QwenImage21BlockCacheT8.execute(tiny_model(True), start_percent=0, end_percent=1)[0]
        patched = nodes.QwenImage21SpectrumT8.execute(patched, start_percent=0, end_percent=1)[0]
        _, state = run_sampling(patched, [0.8, 0.7, 0.6, 0.5])
        self.assertEqual((state.cache.full, state.cache.hits), (2, 2))

    def test_no_skip_matches_native_with_reference_kv_enabled(self):
        base = tiny_model()
        model = base.model.diffusion_model
        patched = nodes.QwenImage21BlockCacheT8.execute(base, residual_diff_threshold=0,
                                                       start_percent=0, end_percent=1)[0]
        model.current_patcher = patched
        model.reset_prefix_cache(True)
        try:
            got, state = run_sampling(patched, [0.7, 0.6], edit=True, batch=2)
            self.assertTrue(model.prefix_cache.filled(3))
            model.reset_prefix_cache(True)
            torch.manual_seed(24)
            x, context = torch.randn(2, 4, 2, 3), torch.randn(2, 4, 8)
            refs = [torch.randn(2, 4, 2, 2)]
            for i, sigma in enumerate([0.7, 0.6]):
                expected = model(x.clone(), torch.full((2,), sigma), context, refs, [2], {})
                torch.testing.assert_close(got[i], expected, rtol=0, atol=0)
            self.assertEqual(state.cache.hits, 0)
        finally:
            model.reset_prefix_cache(False)

    def test_sigma_repeat_and_reversal_refresh(self):
        patched = nodes.QwenImage21BlockCacheT8.execute(tiny_model(True), start_percent=0, end_percent=1)[0]
        _, state = run_sampling(patched, [0.7, 0.7, 0.8])
        self.assertEqual(state.cache.hits, 0)

    def test_no_uuid_disables_reuse(self):
        patched = nodes.QwenImage21BlockCacheT8.execute(tiny_model(True), start_percent=0, end_percent=1)[0]
        _, state = run_sampling(patched, [0.7, 0.6], uuids=[None, None])
        # UUID 'None' must not be accepted as conditioning identity.
        self.assertEqual(state.cache.hits, 0)

    def test_sampling_window_forces_refresh(self):
        patched = nodes.QwenImage21BlockCacheT8.execute(tiny_model(True), start_percent=0.2, end_percent=0.8)[0]
        _, state = run_sampling(patched, [0.99, 0.9, 0.7, 0.1])
        self.assertEqual(state.cache.hits, 1)
        self.assertEqual(state.cache.full, 3)

    def test_spectrum_forecast_and_shared_cap_in_both_orders(self):
        for order in (0, 1):
            patched = tiny_model(True)
            calls = [lambda m: nodes.QwenImage21BlockCacheT8.execute(m, residual_diff_threshold=0, start_percent=0, end_percent=1)[0],
                     lambda m: nodes.QwenImage21SpectrumT8.execute(m, history=3, degree=1, start_percent=0, end_percent=1)[0]]
            if order:
                calls.reverse()
            for call in calls:
                patched = call(patched)
            _, state = run_sampling(patched, [0.8, 0.7, 0.6, 0.5, 0.4])
            self.assertEqual(state.cache.forecasts, 1)
            self.assertEqual(state.cache.full, 4)
            self.assertEqual(len(patched.get_all_wrappers(pe.WrappersMP.DIFFUSION_MODEL)), 1)

    def test_foreign_patch_late_conflict_and_wrapper(self):
        patched = nodes.QwenImage21BlockCacheT8.execute(tiny_model(True))[0]
        patched.set_model_patch_replace(object(), "dit", "single_block", 1)
        with self.assertRaisesRegex(ValueError, "conflicts"):
            run_sampling(patched, [0.5])
        base = tiny_model(True)
        base.model_options["model_function_wrapper"] = object()
        with self.assertRaisesRegex(ValueError, "external model wrapper"):
            nodes.QwenImage21SpectrumT8.execute(base)

    def test_external_transformer_options_wrapper_rejected(self):
        model = tiny_model(True)
        pe.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, "spectrum_qwen21", lambda *a: None,
                                model.model_options, is_model_options=True)
        with self.assertRaisesRegex(ValueError, "external diffusion wrapper"):
            nodes.QwenImage21BlockCacheT8.execute(model)
        patched = nodes.QwenImage21BlockCacheT8.execute(tiny_model(True))[0]
        options = patched.model_options["transformer_options"]
        pe.add_wrapper_with_key(pe.WrappersMP.DIFFUSION_MODEL, "spectrum_qwen21", lambda *a: None, options)
        with self.assertRaisesRegex(ValueError, "external diffusion wrapper"):
            runtime.validate_patches(options, 3)

    def test_exception_releases_runtime_and_restores_options(self):
        patched = nodes.QwenImage21BlockCacheT8.execute(tiny_model(True))[0]
        _, state = run_sampling(patched, [0.5], error=True)
        self.assertEqual(state.cache.streams, {})

    def test_scheduled_lora_uses_real_forwards(self):
        patched = nodes.QwenImage21BlockCacheT8.execute(tiny_model(True))[0]
        patched.hook_patches["scheduled"] = object()
        _, state = run_sampling(patched, [0.7, 0.6])
        self.assertIsNone(state.cache)
        self.assertEqual([b.calls for b in patched.model.diffusion_model.transformer_blocks], [2, 2, 2])

    def test_sage_and_sol_nodes_keep_native_prefix_cache_path(self):
        base = tiny_model()
        sage = nodes.QwenImage21SageAttentionT8.execute(base)[0]
        both = nodes.QwenImage21SolAttentionT8.execute(sage, enabled=True)[0]
        self.assertNotIn("patches_replace", both.model_options["transformer_options"])
        outputs, state = run_sampling(both, [0.6], edit=True)
        self.assertIsNone(state.cache)
        self.assertEqual(outputs[0].shape, (1, 4, 2, 3))

    def test_sol_is_opt_in_and_can_be_disabled_after_install(self):
        base = tiny_model()
        disabled = nodes.QwenImage21SolAttentionT8.execute(base)[0]
        self.assertIsNone(disabled.model_options["transformer_options"][runtime.KEY]["sol"])
        enabled = nodes.QwenImage21SolAttentionT8.execute(base, enabled=True)[0]
        disabled = nodes.QwenImage21SolAttentionT8.execute(enabled, enabled=False)[0]
        self.assertIsNone(disabled.model_options["transformer_options"][runtime.KEY]["sol"])

    def test_sol_window_rejects_mixed_batch_sigmas(self):
        config = nodes.SolConfig()
        model = tiny_model().model.diffusion_model
        seen = []

        def forward(*args, **kwargs):
            seen.append(args[5]["optimized_attention_override"])
            return args[0]

        executor = pe.WrapperExecutor.new_class_executor(forward, model, [runtime.diffusion_wrapper])
        for sigmas, expected in (([0.5, 0.99], None), ([0.5, 0.5], config.start_percent), ([0.99, 0.99], None)):
            options = {runtime.KEY: {"sol": config}, "sigmas": torch.tensor(sigmas),
                       runtime.RUNTIME: runtime.SamplingRuntime(None, (0.85, 0.15), {})}
            with self.subTest(sigmas=sigmas), patch.object(runtime, "make_override", return_value="test") as override:
                executor.execute(torch.zeros(2, 4, 2, 3), torch.tensor(sigmas), torch.zeros(2, 4, 8), transformer_options=options)
                self.assertEqual(override.call_args.args[6], expected)
        self.assertEqual(seen, ["test"] * 3)

    def test_native_kitchen_backend_preserved_in_both_orders(self):
        from comfy_extras.nodes_model_advanced import ModelAttentionBackend
        for reverse in (False, True):
            base = tiny_model()
            if reverse:
                base = nodes.QwenImage21BlockCacheT8.execute(base)[0]
                base = ModelAttentionBackend.execute(base, "comfy kitchen attention")[0]
            else:
                base = ModelAttentionBackend.execute(base, "comfy kitchen attention")[0]
                prior = base.model_options["transformer_options"]["optimized_attention_override"]
                base = nodes.QwenImage21BlockCacheT8.execute(base)[0]
                self.assertIs(base.model_options["transformer_options"]["optimized_attention_override"], prior)
            self.assertIsNotNone(base.model_options["transformer_options"]["optimized_attention_override"])
            self.assertEqual(len(base.get_all_wrappers(pe.WrappersMP.DIFFUSION_MODEL)), 1)

    def test_upstream_attention_override_survives_cache(self):
        base = tiny_model()
        seen = []

        def upstream(func, *args, **kwargs):
            seen.append(kwargs.get("mask") is not None)
            return func(*args, **kwargs)

        base.model_options["transformer_options"]["optimized_attention_override"] = upstream
        patched = nodes.QwenImage21BlockCacheT8.execute(base, residual_diff_threshold=0)[0]
        run_sampling(patched, [0.6], edit=True)
        self.assertIn(True, seen)
        self.assertIn(False, seen)

    def test_prefetch_cleanup_preserves_unrelated_queue(self):
        block, other = object(), object()
        mine = [(None, (block, [object()])), None]
        unrelated = [(None, (other, [object()])), None]
        model = types.SimpleNamespace(transformer_blocks=[block])
        with patch.object(runtime.prefetch, "PREFETCH_QUEUES", [unrelated, mine]), \
             patch.object(runtime.prefetch, "cleanup_prefetched_modules") as cleanup, \
             patch.object(runtime.prefetch, "prefetch_queue_pop") as pop, \
             patch.object(runtime.prefetch, "malloc_graph_end") as end:
            runtime.finish_prefetch(model, torch.device("cpu"))
            self.assertEqual(mine, [None])
            self.assertEqual(len(unrelated), 2)
            self.assertEqual(cleanup.call_count, 1)
            pop.assert_called_once()
            end.assert_called_once()


class ForecastTests(unittest.TestCase):
    def test_fp16_cache_reconstruction_overflow_falls_back(self):
        state = cache.CacheRuntime(cache.CacheConfig(start_percent=0, end_percent=1), None, Sampling())
        stream = state.stream(("fp16",), 0.8)
        indicator = torch.ones(1, 1, 1)
        target = torch.full((1, 2, 2), 40000, dtype=torch.float16)
        state.store(stream, indicator, 0.8, target, torch.zeros_like(target))
        replay, kind = state.replay(stream, indicator, 0.7, target)
        self.assertIsNone(replay)
        self.assertIsNone(kind)
        self.assertEqual((state.hits, stream.consecutive), (0, 0))

    def test_fp16_residual_overflow_is_not_stored(self):
        state = cache.CacheRuntime(cache.CacheConfig(), None, Sampling())
        stream = state.stream(("fp16",), 0.8)
        target = torch.full((1, 2, 2), 40000, dtype=torch.float16)
        state.store(stream, torch.ones(1, 1, 1), 0.8, target, -target)
        self.assertEqual(stream.history, [])
        self.assertIsNone(stream.indicator)

    def test_spectrum_checks_result_after_low_precision_conversion(self):
        for value, target_value in ((80000, 0), (40000, 40000)):
            with self.subTest(value=value, target=target_value):
                state = cache.CacheRuntime(None, cache.SpectrumConfig(history=3, degree=1, start_percent=0, end_percent=1), Sampling())
                stream = state.stream(("fp16",), 0.8)
                indicator = torch.ones(1, 1, 1)
                for sigma in (0.8, 0.7, 0.6):
                    tail = torch.full((1, 2, 2), value / 2, dtype=torch.float16)
                    state.store(stream, indicator, sigma, tail, torch.zeros_like(tail))
                target = torch.full_like(tail, target_value)
                with patch.object(cache, "forecast_weights", return_value=[0, 0, 2]):
                    replay, kind = state.replay(stream, indicator, 0.5, target)
                self.assertIsNone(replay)
                self.assertIsNone(kind)
                self.assertEqual((state.forecasts, stream.consecutive), (0, 0))

    def test_default_quadratic_allows_next_sampling_step(self):
        weights = cache.forecast_weights([0.9, 0.8, 0.7, 0.6], 0.5, 2, 0.01)
        self.assertIsNotNone(weights)
        self.assertAlmostEqual(sum(weights), 1.0, places=8)
        got = sum(w * (2 + s + 3 * s * s) for w, s in zip(weights, [0.9, 0.8, 0.7, 0.6]))
        self.assertAlmostEqual(got, 3.25, delta=0.01)  # ridge intentionally biases the fit

    def test_small_ridge_recovers_quadratic(self):
        sigmas = [0.9, 0.8, 0.7, 0.6]
        weights = cache.forecast_weights(sigmas, 0.55, 2, 1e-8)
        got = sum(w * (2 + s + 3 * s * s) for w, s in zip(weights, sigmas))
        self.assertAlmostEqual(got, 2 + 0.55 + 3 * 0.55 ** 2, places=5)

    def test_far_extrapolation_rejected(self):
        self.assertIsNone(cache.forecast_weights([0.9, 0.8, 0.7, 0.6], 0.01, 2, 0.01))

    def test_batch_worst_row_forces_full(self):
        previous = torch.ones(2, 2, 2)
        current = previous.clone()
        current[1] = 3
        self.assertEqual(cache.CacheRuntime.difference(current, previous), 2)

    def test_memory_budget_evicts_and_storage_is_owned(self):
        state = cache.CacheRuntime(cache.CacheConfig(max_cache_mb=1), None, Sampling())
        stream = state.stream(("a",), 0.5)
        original = torch.ones(1, 8, 8)
        state.store(stream, torch.ones(1, 1, 1), 0.5, original, torch.zeros_like(original))
        self.assertNotEqual(stream.history[0][1].data_ptr(), original.data_ptr())
        state.budget = 1
        state.trim()
        self.assertEqual(state.bytes(), 0)


if __name__ == "__main__":
    unittest.main()
