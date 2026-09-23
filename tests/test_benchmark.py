from contextlib import asynccontextmanager
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import aiohttp

import benchmark_native as benchmark


@asynccontextmanager
async def context(value):
    yield value


class Response:
    def __init__(self, value, status=200):
        self.value = value
        self.status = status

    async def json(self):
        return self.value


class Socket:
    def __init__(self, events, disconnect=False):
        self.events = events
        self.disconnect = disconnect

    async def __aiter__(self):
        for event in self.events:
            yield SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(event))
        if self.disconnect:
            raise ConnectionError("test websocket disconnect")


class BenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def run_fake(self, events, disconnect=False, history=None, busy=False, status=200):
        args = SimpleNamespace(timeout=5, modes="baseline", url="http://127.0.0.1:1", size=512, steps=2)
        queue = {"queue_running": [1] if busy else [], "queue_pending": []}
        session = SimpleNamespace(
            get=Mock(side_effect=[context(Response(queue)), context(Response({"test": history}))]),
            post=Mock(return_value=context(Response({"prompt_id": "test"}, status))),
            ws_connect=Mock(return_value=context(Socket(events, disconnect))),
        )
        state = {"pending": False}
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(benchmark, "ROOT", Path(folder)), \
             patch.object(benchmark, "graph", return_value={}), \
             patch.object(benchmark.aiohttp, "ClientSession", return_value=context(session)), \
             patch("builtins.print"):
            error = None
            try:
                await benchmark.run(args, state)
            except (ConnectionError, RuntimeError) as caught:
                error = caught
        return state, error, session

    async def test_disconnect_retains_pending_state(self):
        state, error, _ = await self.run_fake([], disconnect=True)
        self.assertIsInstance(error, ConnectionError)
        self.assertTrue(state["pending"])

    async def test_busy_queue_never_submits(self):
        state, error, session = await self.run_fake([], busy=True)
        self.assertIsInstance(error, RuntimeError)
        self.assertFalse(state["pending"])
        session.post.assert_not_called()

    async def test_rejected_submission_does_not_hold_lock(self):
        state, error, _ = await self.run_fake([], status=400)
        self.assertIsInstance(error, RuntimeError)
        self.assertFalse(state["pending"])

    async def test_success_requires_sampler_and_terminal_history(self):
        events = [{"type": "executing", "data": {"prompt_id": "test", "node": node}} for node in ("100", None)]
        history = {"status": {"completed": True, "status_str": "success"}, "outputs": {}}
        state, error, _ = await self.run_fake(events, history=history)
        self.assertIsNone(error)
        self.assertFalse(state["pending"])
        history["status"] = {"completed": False, "status_str": "running"}
        state, error, _ = await self.run_fake(events, history=history)
        self.assertIsInstance(error, RuntimeError)
        self.assertTrue(state["pending"])
        history["status"]["status_str"] = "error"
        state, error, _ = await self.run_fake(events, history=history)
        self.assertIsInstance(error, RuntimeError)
        self.assertFalse(state["pending"])

    async def test_execution_cache_is_not_a_speed_measurement(self):
        events = [{"type": "executing", "data": {"prompt_id": "test", "node": None}}]
        state, error, _ = await self.run_fake(events)
        self.assertIsInstance(error, RuntimeError)
        self.assertIn("not a valid benchmark", str(error))
        self.assertTrue(state["pending"])

    def test_cli_rejects_unknown_and_multiple_modes_without_server(self):
        for mode in ("blok", "spectrum-foo", "baseline,block"):
            result = subprocess.run([sys.executable, str(Path(benchmark.__file__)), "--modes", mode],
                                    capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 2)
            self.assertIn("invalid choice", result.stderr)


if __name__ == "__main__":
    unittest.main()
