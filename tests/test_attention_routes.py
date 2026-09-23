import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from test_runtime import ROOT, nodes, run_sampling, runtime, tiny_model
from qwen21_t8 import attention


def tensors(tokens=1024, dtype=torch.bfloat16, device="cuda:0", shaped=False):
    def tensor(length):
        return SimpleNamespace(shape=(1, 32, length, 128) if shaped else (1, length, 4096),
                               device=torch.device(device), dtype=dtype, requires_grad=False)
    return tensor(tokens), tensor(tokens + 128), tensor(tokens + 128)


class AttentionRouteTests(unittest.TestCase):
    def test_hybrid_canvases_keep_connections_and_tested_settings(self):
        for variant, cache_mode in (("Hybrid_Edit", 0), ("Hybrid_NoCache_Edit", 4)):
            workflow = json.loads((ROOT / "workflows" / f"Qwen21_T8_1024_{variant}.json").read_text(encoding="utf-8"))
            by_id = {node["id"]: node for node in workflow["nodes"]}
            by_type = {node["type"]: node for node in workflow["nodes"]}
            for link, source, source_slot, dest, dest_slot, kind in workflow["links"]:
                self.assertIn(link, by_id[source]["outputs"][source_slot]["links"])
                self.assertEqual(by_id[dest]["inputs"][dest_slot]["link"], link)
            sage = by_type["QwenImage21SageAttentionT8"]
            self.assertEqual((sage["mode"], sage["widgets_values_named"]["backend_mode"]), (0, "sage_kitchen"))
            for kind in ("QwenImage21BlockCacheT8", "QwenImage21SpectrumT8"):
                self.assertEqual(by_type[kind]["mode"], cache_mode)
                self.assertEqual(by_type[kind]["widgets_values_named"]["threshold_mode"], "constant")
            self.assertEqual(by_type["ModelAttentionBackend"]["mode"], 4)
            self.assertEqual(by_type["QwenImage21SolAttentionT8"]["mode"], 4)
            self.assertFalse(by_type["QwenImage21SolAttentionT8"]["widgets_values_named"]["enabled"])
            sampler = by_type["KSampler"]["widgets_values_named"]
            self.assertEqual((sampler["seed"], sampler["steps"], sampler["cfg"]), (42, 40, 1))
            links = {link[0]: link for link in workflow["links"]}
            for kind, socket, upstream in (("TextEncodeQwenImage21", "vae", "VAELoader"),
                                            ("TextEncodeQwenImage21", "images.image_1", "LoadImage"),
                                            ("VAEDecode", "vae", "VAELoader")):
                value = next(item for item in by_type[kind]["inputs"] if item["name"] == socket)
                self.assertEqual(by_id[links[value["link"]][1]]["type"], upstream)

    def test_hybrid_routes_only_long_unmasked_native_calls(self):
        for shaped in (False, True):
            for dtype in (torch.float16, torch.bfloat16):
                for tokens, mask, low, selected in ((1024, None, True, "kitchen"),
                                                    (1023, None, True, "sage"),
                                                    (4096, object(), True, "sage"),
                                                    (4096, None, False, "sage")):
                    stats = {}
                    override = attention.make_override(None, None, "sage_kitchen", None, 4096, 8192, None, stats)
                    q, k, v = tensors(tokens, dtype, shaped=shaped)
                    with patch.object(attention.ck, "int8_attention_is_available", return_value=True), \
                         patch.object(attention.native_attention, "attention_sage") as sage, \
                         patch.object(attention.native_attention, "attention_comfy_kitchen_int8") as kitchen:
                        result = override(Mock(), q, k, v, 32, mask=mask, skip_reshape=shaped,
                                          skip_output_reshape=True, low_precision_attention=low, scale=0.07)
                        called, unused = (kitchen, sage) if selected == "kitchen" else (sage, kitchen)
                        self.assertIs(result, called.return_value)
                        self.assertEqual(stats, {selected: 1})
                        unused.assert_not_called()
                        self.assertEqual(called.call_args.args, (q, k, v, 32))
                        self.assertIs(called.call_args.kwargs["mask"], mask)
                        self.assertTrue(called.call_args.kwargs["_inside_attn_wrapper"])
                        self.assertTrue(called.call_args.kwargs["skip_output_reshape"])
                        self.assertEqual(called.call_args.kwargs["scale"], 0.07)

    def test_unavailable_kitchen_preserves_sage(self):
        override = attention.make_override(None, None, "sage_kitchen", None, 4096, 8192, None, {})
        with patch.object(attention.ck, "int8_attention_is_available", return_value=False), \
             patch.object(attention.native_attention, "attention_sage") as sage, \
             patch.object(attention.native_attention, "attention_comfy_kitchen_int8") as kitchen:
            self.assertIs(override(Mock(), *tensors(), 32), sage.return_value)
            kitchen.assert_not_called()

    def test_plain_sage_does_not_probe_or_override_with_kitchen(self):
        override = attention.make_override(None, None, True, None, 4096, 8192, None, {})
        with patch.object(attention.ck, "int8_attention_is_available") as available, \
             patch.object(attention.native_attention, "attention_sage") as sage:
            self.assertIs(override(Mock(), *tensors(), 32), sage.return_value)
            available.assert_not_called()

    def test_unsupported_inputs_preserve_upstream_without_probe(self):
        previous = Mock()
        override = attention.make_override(None, previous, "sage_kitchen", None, 4096, 8192, None, {})
        cases = [tensors(device="cpu"), tensors(dtype=torch.float32), tensors()]
        cases[-1][0].requires_grad = True
        with patch.object(attention.ck, "int8_attention_is_available") as available:
            for values in cases:
                self.assertIs(override(Mock(), *values, 32), previous.return_value)
            available.assert_not_called()

    def test_node_mode_default_and_optional_schema(self):
        cls = nodes.QwenImage21SageAttentionT8
        self.assertTrue(cls.define_schema().inputs[-1].optional)
        with patch.object(nodes.native_attention, "SAGE_ATTENTION_IS_AVAILABLE", True):
            old = cls.execute(tiny_model())[0]
            new = cls.execute(tiny_model(), backend_mode="sage_kitchen")[0]
            self.assertIs(old.model_options["transformer_options"][runtime.KEY]["sage"], True)
            self.assertEqual(new.model_options["transformer_options"][runtime.KEY]["sage"], "sage_kitchen")
            with self.assertRaisesRegex(ValueError, "backend_mode"):
                cls.execute(tiny_model(), backend_mode="unknown")

    def test_hybrid_sampling_cleanup_and_cache_composition(self):
        with patch.object(nodes.native_attention, "SAGE_ATTENTION_IS_AVAILABLE", True):
            model = nodes.QwenImage21SageAttentionT8.execute(tiny_model(True), backend_mode="sage_kitchen")[0]
        model = nodes.QwenImage21BlockCacheT8.execute(model, start_percent=0, end_percent=1)[0]
        for _ in range(2):
            _, state = run_sampling(model, [0.7, 0.6, 0.5], edit=True)
            self.assertEqual(state.cache.hits, 2)
            self.assertEqual(state.stats, {})


if __name__ == "__main__":
    unittest.main()
