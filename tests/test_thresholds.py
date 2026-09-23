import unittest
from unittest.mock import patch

import torch

from test_runtime import Sampling, cache, nodes, run_sampling, runtime, tiny_model


class ThresholdTests(unittest.TestCase):
    def test_constant_mode_preserves_old_thresholds(self):
        block = cache.CacheConfig(threshold=0.03, late_threshold=1)
        spectrum = cache.SpectrumConfig(guard_threshold=0.08, late_threshold=0)
        state = cache.CacheRuntime(block, spectrum, Sampling())
        self.assertEqual(state.splits, {})
        for sigma in (0.95, 0.7, 0.5, 0.2, 0.05):
            self.assertEqual(state.threshold(block, sigma, block.threshold), 0.03)
            self.assertEqual(state.threshold(spectrum, sigma, spectrum.guard_threshold), 0.08)

    def test_split_uses_native_progress_conversion_not_sigma_interpolation(self):
        sampling = Sampling()
        sampling.percent_to_sigma = lambda p: (1 - p) ** 2
        for ratio in (0.25, 0.5, 0.75):
            block = cache.CacheConfig(start_percent=0.15, end_percent=0.85,
                                      threshold_mode="two_stage", split_ratio=ratio)
            state = cache.CacheRuntime(block, None, sampling)
            progress = 0.15 + 0.7 * ratio
            split = sampling.percent_to_sigma(progress)
            self.assertEqual(state.splits[id(block)], split)
            self.assertEqual(state.threshold(block, split + 0.001, 0.2), 0.2)
            self.assertEqual(state.threshold(block, split, 0.2), block.late_threshold)
            self.assertEqual(state.threshold(block, split - 0.001, 0.2), block.late_threshold)

    def test_window_and_split_endpoints(self):
        for ratio in (0, 0.5, 1):
            config = cache.SpectrumConfig(start_percent=0.2, end_percent=0.8,
                                          threshold_mode="two_stage", split_ratio=ratio)
            state = cache.CacheRuntime(None, config, Sampling())
            start, end = state.windows[id(config)]
            self.assertFalse(state.window(config, start + 0.001))
            self.assertTrue(state.window(config, start))
            self.assertFalse(state.window(config, end))
            self.assertFalse(state.window(config, end - 0.001))
            for sigma in (start, end + 0.001):
                if ratio in (0, 1):
                    expected = config.late_threshold if ratio == 0 else config.guard_threshold
                    self.assertEqual(state.threshold(config, sigma, config.guard_threshold), expected)

    def test_combined_nodes_have_independent_splits(self):
        block = cache.CacheConfig(start_percent=0.1, end_percent=0.9, threshold_mode="two_stage", split_ratio=0.25)
        spectrum = cache.SpectrumConfig(start_percent=0.2, end_percent=0.8, threshold_mode="two_stage", split_ratio=0.75)
        state = cache.CacheRuntime(block, spectrum, Sampling())
        for sigma in (0.5, 0.8, 0.5, 0.2, 0.5):
            if sigma == 0.5:
                self.assertEqual(state.threshold(block, sigma, 0.2), block.late_threshold)
                self.assertEqual(state.threshold(spectrum, sigma, 0.2), 0.2)

    def test_both_algorithms_use_early_and_late_thresholds(self):
        for kind in ("block", "spectrum"):
            for early, late in ((0.2, 0.05), (0.05, 0.2)):
                for sigma, threshold in ((0.7, early), (0.3, late)):
                    with self.subTest(kind=kind, early=early, late=late, sigma=sigma):
                        common = dict(start_percent=0, end_percent=1, threshold_mode="two_stage", late_threshold=late)
                        block = cache.CacheConfig(threshold=early, **common) if kind == "block" else None
                        spectrum = cache.SpectrumConfig(guard_threshold=early, **common) if kind == "spectrum" else None
                        state = cache.CacheRuntime(block, spectrum, Sampling())
                        stream = state.stream(("positive",), 0.9)
                        target = torch.ones(1, 2, 2)
                        for s in (0.95, 0.9, 0.85, 0.8):
                            state.store(stream, torch.ones_like(target), s, target, torch.zeros_like(target))
                        with patch.object(cache, "forecast_weights", return_value=[0, 0, 0, 1]):
                            value, method = state.replay(stream, target * 1.1, sigma, target)
                        self.assertEqual(value is not None, threshold == 0.2)
                        self.assertEqual(method, ("cache" if kind == "block" else "spectrum") if threshold == 0.2 else None)

    def test_large_late_threshold_does_not_override_end_window(self):
        config = cache.CacheConfig(start_percent=0.2, end_percent=0.8, threshold_mode="two_stage", late_threshold=1)
        state = cache.CacheRuntime(config, None, Sampling())
        target = torch.ones(1, 2, 2)
        stream = state.stream(("positive",), 0.9)
        state.store(stream, target, 0.9, target, torch.zeros_like(target))
        for sigma in (0.95, 0.1):
            self.assertEqual(state.replay(stream, target, sigma, target), (None, None))

    def test_native_blocks_stop_skipping_after_progress_split(self):
        base = tiny_model(True)
        patched = nodes.QwenImage21BlockCacheT8.execute(
            base, residual_diff_threshold=1, start_percent=0.2, end_percent=0.8,
            max_consecutive_hits=10, threshold_mode="two_stage", split_ratio=0.5, late_threshold=0,
        )[0]
        sigmas = [0.95, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]
        for _ in range(2):
            _, state = run_sampling(patched, sigmas, edit=True)
            self.assertEqual((state.cache.full, state.cache.hits), (6, 3))

    def test_old_node_calls_and_optional_appended_widgets(self):
        for cls, expected, late in ((nodes.QwenImage21BlockCacheT8, 7, 0.03), (nodes.QwenImage21SpectrumT8, 9, 0.08)):
            schema = cls.define_schema()
            inputs = schema.inputs
            self.assertEqual([v.id for v in inputs[expected + 1:]], ["threshold_mode", "split_ratio", "late_threshold"])
            self.assertTrue(all(v.optional for v in inputs[-3:]))
            patched = cls.execute(tiny_model())[0]
            config = patched.model_options["transformer_options"][runtime.KEY]
            selected = config["block" if cls is nodes.QwenImage21BlockCacheT8 else "spectrum"]
            self.assertEqual((selected.threshold_mode, selected.split_ratio, selected.late_threshold), ("constant", 0.5, late))

    def test_native_spectrum_stages_compose_in_both_node_orders(self):
        for reverse in (False, True):
            installers = [
                lambda m: nodes.QwenImage21BlockCacheT8.execute(m, residual_diff_threshold=0, threshold_mode="two_stage", late_threshold=0)[0],
                lambda m: nodes.QwenImage21SpectrumT8.execute(m, history=3, degree=1, guard_threshold=1,
                    start_percent=0, end_percent=1, threshold_mode="two_stage", late_threshold=0)[0],
            ]
            model = tiny_model(True)
            for install in reversed(installers) if reverse else installers:
                model = install(model)
            _, state = run_sampling(model, [0.95, 0.85, 0.75, 0.65, 0.45, 0.35], edit=True)
            self.assertEqual((state.cache.full, state.cache.hits, state.cache.forecasts), (5, 0, 1))


if __name__ == "__main__":
    unittest.main()
