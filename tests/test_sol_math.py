import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import comfy_kitchen as ck
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
CORE = ROOT.parents[1]
sys.path.insert(0, str(CORE))
_argv = sys.argv
try:
    sys.argv = [sys.argv[0], "--cpu"]
    if "qwen21_t8" not in sys.modules:
        _spec = importlib.util.spec_from_file_location(
            "qwen21_t8", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
        )
        _package = importlib.util.module_from_spec(_spec)
        sys.modules[_spec.name] = _package
        _spec.loader.exec_module(_package)
    attention = sys.modules["qwen21_t8.attention"]
finally:
    sys.argv = _argv


class SolMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def tensors(self, prefix, target=32, batch=2, heads=2):
        generator = torch.Generator(device="cpu").manual_seed(42)
        q = torch.randn(batch, target, heads, 128, generator=generator)
        k = torch.randn(batch, prefix + target, heads, 128, generator=generator)
        v = torch.randn(k.shape, generator=generator)
        return q, k, v

    @staticmethod
    def dense(q, k, v, scale=None):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=scale
        ).transpose(1, 2)

    def test_square_adapter_matches_rectangular_sdpa_for_exact_small_cases(self):
        for prefix in (0, 33, 64):
            for scale in (None, 0.07):
                with self.subTest(prefix=prefix, scale=scale), ck.use_backend("eager"):
                    q, k, v = self.tensors(prefix)
                    actual = attention.square_target_attention(q, k, v, 1.0, scale)
                    torch.testing.assert_close(actual, self.dense(q, k, v, scale), atol=1e-6, rtol=1e-5)
                    self.assertEqual(actual.shape, q.shape)
                    self.assertEqual(actual.device.type, "cpu")
                    self.assertEqual(actual.dtype, q.dtype)

    def test_mixed_query_block_is_exact_and_dummy_queries_do_not_change_targets(self):
        prefix = 33
        q, k, v = self.tensors(prefix, target=288, heads=1)
        original = ck.sol_attn
        calls = []

        def with_changed_dummy(padded, keys, values, **kwargs):
            calls.append(kwargs)
            changed = padded.clone()
            changed[:, :prefix] = 100
            return original(changed, keys, values, **kwargs)

        with ck.use_backend("eager"):
            baseline = attention.square_target_attention(q, k, v, 0.8)
            with patch.object(ck, "sol_attn", side_effect=with_changed_dummy):
                changed = attention.square_target_attention(q, k, v, 0.8)
        self.assertEqual(calls[0]["sink_blocks"], [0, 1])
        self.assertEqual(calls[0]["sink_q"], [0, 1])
        torch.testing.assert_close(changed, baseline, atol=0, rtol=0)
        boundary_targets = 64 - prefix
        torch.testing.assert_close(
            baseline[:, :boundary_targets], self.dense(q, k, v)[:, :boundary_targets],
            atol=1e-6, rtol=1e-5,
        )

    def test_full_prefix_blocks_do_not_force_target_query_blocks_dense(self):
        q, k, v = self.tensors(64)
        with patch.object(ck, "sol_attn", return_value=torch.zeros_like(k)) as kernel:
            output = attention.square_target_attention(q, k, v, 1.0)
        self.assertEqual(kernel.call_args.kwargs["sink_blocks"], [0, 1])
        self.assertEqual(kernel.call_args.kwargs["sink_q"], [0, 0])
        self.assertEqual(output.shape, q.shape)

    def test_cpu_masked_reference_and_precision_calls_preserve_previous_backend(self):
        config = attention.SolConfig(min_tokens=64)
        marker = object()
        previous = Mock(return_value=marker)
        base = Mock()
        stats = {}
        model = SimpleNamespace(transformer_blocks=[])
        override = attention.make_override(model, previous, False, config, 64, 97, 0.5, stats)
        cases = [
            (64, None, {}),
            (64, torch.ones(64, 97, dtype=torch.bool), {}),
            (33, None, {}),
            (64, None, {"low_precision_attention": False}),
        ]
        with patch.object(ck, "sol_attn") as kernel:
            for tokens, mask, options in cases:
                with self.subTest(tokens=tokens, masked=mask is not None, options=options):
                    q = torch.zeros(2, tokens, 128)
                    k = torch.zeros(2, 97, 128)
                    v = torch.ones_like(k)
                    result = override(base, q, k, v, 1, mask=mask, **options)
                    self.assertIs(result, marker)
                    self.assertIs(previous.call_args.args[1], q)
                    self.assertIs(previous.call_args.kwargs["mask"], mask)
                    self.assertTrue(previous.call_args.kwargs["_inside_attn_wrapper"])
                    for key, value in options.items():
                        self.assertEqual(previous.call_args.kwargs[key], value)
            kernel.assert_not_called()
        base.assert_not_called()
        self.assertEqual(stats["sol_dense"], len(cases))

    def test_declined_call_preserves_native_preferred_backend(self):
        marker = object()
        preferred = Mock(return_value=marker)
        model = SimpleNamespace(transformer_blocks=[SimpleNamespace(
            attn=SimpleNamespace(comfy_attention=SimpleNamespace(function=preferred))
        )])
        base = Mock()
        override = attention.make_override(model, None, False, attention.SolConfig(), 32, 65, None, {})
        q, k, v = self.tensors(33, heads=1)
        output = override(base, q.flatten(2), k.flatten(2), v.flatten(2), 1,
                          transformer_options={"block_index": 0})
        self.assertIs(output, marker)
        preferred.assert_called_once()
        base.assert_not_called()

    def test_mask_reference_and_dtype_guards_decline_before_gpu_kernel_probe(self):
        def metadata(tokens, dtype=torch.bfloat16):
            return SimpleNamespace(shape=(2, tokens, 128), dtype=dtype,
                                   device=torch.device("cuda:0"), requires_grad=False)

        previous = Mock(return_value=object())
        override = attention.make_override(
            SimpleNamespace(transformer_blocks=[]), previous, False,
            attention.SolConfig(min_tokens=64), 64, 97, 0.5, {},
        )
        cases = [
            (metadata(64), torch.ones(64, 97, dtype=torch.bool), {}),
            (metadata(33), None, {}),
            (metadata(64, torch.float32), None, {}),
            (metadata(64), None, {"low_precision_attention": False}),
        ]
        with patch.object(ck, "sol_attn_is_available") as available, patch.object(ck, "sol_attn") as kernel:
            for q, mask, options in cases:
                result = override(Mock(), q, metadata(97), metadata(97), 1, mask=mask, **options)
                self.assertIs(result, previous.return_value)
            available.assert_not_called()
            kernel.assert_not_called()

    def test_unavailable_compiled_backend_uses_previous_without_allocating_cuda(self):
        previous = Mock(return_value=object())
        override = attention.make_override(
            SimpleNamespace(transformer_blocks=[]), previous, False,
            attention.SolConfig(min_tokens=64), 64, 97, 0.5, {},
        )
        q = SimpleNamespace(shape=(2, 64, 128), dtype=torch.bfloat16,
                            device=torch.device("cuda:0"), requires_grad=False)
        k = SimpleNamespace(shape=(2, 97, 128), dtype=torch.bfloat16,
                            device=torch.device("cuda:0"), requires_grad=False)
        with patch.object(ck, "sol_attn_is_available", return_value=False) as available, patch.object(ck, "sol_attn") as kernel:
            result = override(Mock(), q, k, k, 1)
            self.assertIs(result, previous.return_value)
            available.assert_called_once_with(torch.device("cuda:0"))
            kernel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
