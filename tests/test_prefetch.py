import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import torch

from test_runtime import nodes, run_sampling, runtime, tiny_model
from comfy.ldm.qwen_image21 import model as qwen


def writable_weight():
    # Exercise Core's actual non-iterable class when available (0.42+).
    cls = getattr(runtime.comfy.ops, "WritableWeight", None)
    return cls((1,), torch.float32) if cls else torch.nn.Module()


class PrefetchTests(unittest.TestCase):
    def finish(self, blocks, queues):
        model = SimpleNamespace(transformer_blocks=blocks)
        with patch.object(runtime.prefetch, "PREFETCH_QUEUES", queues), \
             patch.object(runtime.prefetch, "cleanup_prefetched_modules") as cleanup, \
             patch.object(runtime.prefetch, "prefetch_queue_pop") as pop, \
             patch.object(runtime.prefetch, "malloc_graph_end") as end, \
             patch.object(runtime.comfy.model_management, "current_stream", return_value="compute"):
            runtime.finish_prefetch(model, torch.device("cpu"))
        pop.assert_called_once_with(None, torch.device("cpu"), None, malloc_scope="block")
        end.assert_called_once_with()
        return cleanup

    def test_mixed_queue_cleans_only_prefetched_entries(self):
        blocks = [torch.nn.Module() for _ in range(4)]
        stream = Mock()
        modules = [writable_weight()]
        queue = [(stream, (blocks[0], modules)),
                 (blocks[1], writable_weight()),
                 (blocks[2], writable_weight(), writable_weight()),
                 (blocks[3],), None]
        cleanup = self.finish(blocks, [queue])
        self.assertEqual(queue, [None])
        stream.wait_stream.assert_called_once_with("compute")
        cleanup.assert_called_once_with(blocks[0], modules)

    def test_unconsumed_groups_still_identify_the_model_queue(self):
        block = torch.nn.Module()
        for group in ((block,), (block, writable_weight()),
                      (block, writable_weight(), writable_weight()),
                      [block, writable_weight()], block):
            with self.subTest(group=type(group).__name__):
                queue = [None, group, None]
                cleanup = self.finish([block], [queue])
                self.assertEqual(queue, [None])
                cleanup.assert_not_called()

    def test_unrelated_grouped_queues_are_preserved(self):
        block, other = torch.nn.Module(), torch.nn.Module()
        stream = Mock()
        modules = [writable_weight()]
        mine = [(None, (block, modules)), (block, writable_weight()), None]
        unrelated = [(stream, (other, [writable_weight()])), (other,), None]
        before = list(unrelated)
        cleanup = self.finish([block], [mine, unrelated])
        self.assertEqual(mine, [None])
        self.assertEqual(unrelated, before)
        stream.wait_stream.assert_not_called()
        cleanup.assert_called_once_with(block, modules)

    def test_legacy_entries_keep_stream_waits_and_cleanup(self):
        blocks = [torch.nn.Module() for _ in range(4)]
        streams = [Mock(), Mock()]
        modules = [writable_weight()]
        queue = [(streams[0], (blocks[0], modules)),
                 (streams[1], (blocks[1], None)),
                 (None, (blocks[2], [])), blocks[3], None]
        cleanup = self.finish(blocks, [queue])
        self.assertEqual(queue, [None])
        for stream in streams:
            stream.wait_stream.assert_called_once_with("compute")
        self.assertEqual(cleanup.call_args_list, [call(blocks[0], modules), call(blocks[2], [])])

    def test_no_matching_queue_and_disabled_prefetch(self):
        block, other = torch.nn.Module(), torch.nn.Module()
        for queues in ([], [[None]], [[None, (other, writable_weight()), None]]):
            with self.subTest(queues=len(queues)):
                before = [list(queue) for queue in queues]
                cleanup = self.finish([block], queues)
                self.assertEqual(queues, before)
                cleanup.assert_not_called()

    def test_only_innermost_matching_queue_is_finished(self):
        block = torch.nn.Module()
        outer = [None, (block, writable_weight()), None]
        inner = [(None, (block, [])), (block, writable_weight()), None]
        cleanup = self.finish([block], [outer, inner])
        self.assertEqual(len(outer), 3)
        self.assertEqual(inner, [None])
        cleanup.assert_called_once_with(block, [])

    @unittest.skipUnless(hasattr(qwen, "PrefixCacheSlot"), "Core predates WritableWeight prefix caching")
    def test_native_prefix_cache_hits_finish_grouped_queues(self):
        patched = nodes.QwenImage21BlockCacheT8.execute(
            tiny_model(), residual_diff_threshold=1, start_percent=0, end_percent=1,
            threshold_mode="constant",
        )[0]
        model = patched.model.diffusion_model
        model.current_patcher = patched
        model.reset_prefix_cache(True)
        # Four text tokens + four reference tokens, with native writable K/V storage.
        slot = qwen.PrefixCacheSlot((1, 2, 8, 1, 128), 3, torch.float32, "default")
        slot.to_empty(device="cpu")
        queues = []

        def make_queue(units, device, options):
            self.assertIsInstance(units[0][1], runtime.comfy.ops.WritableWeight)
            queue = [None] + units + [None]
            queues.append(queue)
            return queue

        try:
            with patch.object(runtime.prefetch, "PREFETCH_QUEUES", queues), \
                 patch.object(runtime.prefetch, "make_prefetch_queue", side_effect=make_queue), \
                 patch.object(runtime.prefetch, "pin_modules", return_value=(None, False)):
                outputs, state = run_sampling(patched, [0.7, 0.6, 0.5], edit=True,
                                              prefix_cache={torch.device("cpu"): [slot]})
            self.assertEqual(state.cache.hits, 2)
            self.assertIsNotNone(slot.key)
            self.assertEqual(queues, [[None], [None], [None]])
            self.assertTrue(all(torch.isfinite(out).all() for out in outputs))
            self.assertEqual(outputs[-1].shape, (1, 4, 2, 3))
        finally:
            model.reset_prefix_cache(False)


if __name__ == "__main__":
    unittest.main()
