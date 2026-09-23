"""Opt-in real-weight API benchmark against a dedicated local ComfyUI server.

Uses native loaders, text conditioning, KSampler and VAE. Does not download models.
Run sequentially; execution-cache hits are rejected as invalid timing samples.
"""
# ruff: noqa: T201
import argparse
import asyncio
import json
from pathlib import Path
import time
import uuid

import aiohttp


ROOT = Path(__file__).resolve().parents[1]
MODES = ("baseline", "block", "spectrum", "combined", "sol", "kitchen", "kitchen-block",
         "kitchen-spectrum", "kitchen-combined", "kitchen-sol", "sage", "sage-block",
         "sage-spectrum", "sage-combined", "sage-sol")
DEFAULT_PROMPT = ('A studio photograph of a small red ceramic teapot on a pale wooden table, '
                  'a white card next to it clearly reads "QWEN 2.1", soft window lighting, '
                  'a green plant in the background, realistic glaze and shadows, clean composition.')


def graph(args, mode, nonce):
    prompt = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": args.model, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": args.clip, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": args.vae}},
        "4": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "clip": ["2", 0], "prompt": args.prompt, "negative_prompt": "", "resolution": args.size}},
    }
    current = ["1", 0]
    if args.reference:
        prompt["5"] = {"class_type": "LoadImage", "inputs": {"image": args.reference}}
        prompt["4"]["inputs"].update(vae=["3", 0], image_1=["5", 0])

    def add(node_type, **inputs):
        nonlocal current
        key = str(10 + len(prompt))
        prompt[key] = {"class_type": node_type, "inputs": {"model": current, **inputs}}
        current = [key, 0]

    if "kitchen" in mode:
        add("ModelAttentionBackend", attention="comfy kitchen attention")
    if "sage" in mode:
        add("QwenImage21SageAttentionT8")
    if "sol" in mode:
        add("QwenImage21SolAttentionT8", tau=args.tau, min_tokens=args.sol_min_tokens,
            start_percent=0.15, end_percent=0.85, enabled=True)
    if "block" in mode or "combined" in mode:
        add("QwenImage21BlockCacheT8", residual_diff_threshold=args.threshold,
            start_percent=0.10, end_percent=0.85, max_consecutive_hits=2,
            cache_device=args.cache_device, metric_stride=8, max_cache_mb=1024)
    if "spectrum" in mode or "combined" in mode:
        add("QwenImage21SpectrumT8", history=4, degree=2, ridge=0.01, guard_threshold=args.guard,
            start_percent=0.15, end_percent=0.85, max_consecutive_hits=1,
            cache_device=args.cache_device, max_cache_mb=1024)
    prompt["100"] = {"class_type": "KSampler", "inputs": {
        "model": current, "seed": args.seed, "steps": args.steps, "cfg": args.cfg,
        "sampler_name": "euler", "scheduler": "simple", "positive": ["4", 0],
        "negative": ["4", 1], "latent_image": ["4", 2], "denoise": 1.0,
        # Core hashes unknown literal inputs but does not pass them to legacy nodes.
        # This forces actual sampling without changing any numerical input.
        "_benchmark_nonce": nonce,
    }}
    prompt["101"] = {"class_type": "VAEDecode", "inputs": {"samples": ["100", 0], "vae": ["3", 0]}}
    prompt["102"] = {"class_type": "SaveImage", "inputs": {
        "images": ["101", 0], "filename_prefix": f"qwen21_t8_benchmark/{args.size}_{mode}_{nonce[:8]}"}}
    return prompt


async def run(args, state):
    target = ROOT / "benchmark_results"
    target.mkdir(exist_ok=True)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for mode in args.modes.split(","):
            async with session.get(args.url + "/queue") as response:
                queue = await response.json()
            if queue["queue_running"] or queue["queue_pending"]:
                raise RuntimeError("ComfyUI is busy. Wait until all work finishes; do not overlap tests.")
            nonce = str(uuid.uuid4())
            data = {"mode": mode, "settings": vars(args), "nonce": nonce}
            data["graph"] = graph(args, mode, nonce)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            path = target / f"{stamp}_{args.size}_{mode}_{nonce[:8]}.json"
            async with session.ws_connect(args.url.replace("http", "ws", 1) + "/ws?clientId=" + nonce) as ws:
                state["pending"] = True
                async with session.post(args.url + "/prompt", json={"prompt": data["graph"], "client_id": nonce}) as response:
                    submitted = await response.json()
                    if response.status != 200:
                        state["pending"] = False
                        raise RuntimeError(submitted)
                prompt_id = submitted["prompt_id"]
                print(f"START {mode} {args.size}px {args.steps} steps prompt={prompt_id}", flush=True)
                start = time.perf_counter()
                previous, previous_time = None, start
                durations = {}
                async with asyncio.timeout(args.timeout):
                    async for message in ws:
                        if message.type != aiohttp.WSMsgType.TEXT:
                            continue
                        event = json.loads(message.data)
                        payload = event.get("data", {})
                        if payload.get("prompt_id") != prompt_id:
                            continue
                        if event["type"] in ("execution_error", "execution_interrupted"):
                            data["error"] = payload
                            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                            raise RuntimeError(payload)
                        if event["type"] == "executing":
                            now = time.perf_counter()
                            if previous is not None:
                                durations[previous] = now - previous_time
                            previous, previous_time = payload.get("node"), now
                            if previous is None:
                                break
                if "100" not in durations:
                    raise RuntimeError("KSampler was cached or failed: not a valid benchmark")
                async with session.get(args.url + "/history/" + prompt_id) as response:
                    history = (await response.json())[prompt_id]
                if history["status"]["status_str"] in ("success", "error"):
                    state["pending"] = False
                if not history["status"]["completed"] or history["status"]["status_str"] != "success":
                    raise RuntimeError(history["status"])
                data.update(prompt_id=prompt_id, durations=durations, wall_seconds=time.perf_counter() - start,
                            outputs=history["outputs"], status=history["status"])
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"DONE {mode} sampler={durations['100']:.3f}s wall={data['wall_seconds']:.3f}s result={path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8189")
    parser.add_argument("--model", default="qwen_image_2.1_int8_convrot.safetensors")
    parser.add_argument("--clip", default="qwen3vl_8b_fp8_scaled.safetensors")
    parser.add_argument("--vae", default="qwen_image_2.1_vae_bf16.safetensors")
    parser.add_argument("--modes", default="baseline", choices=MODES, help="One mode per invocation; unload models before starting the next test")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--reference", default="", help="Existing ComfyUI input image filename for edit validation")
    parser.add_argument("--threshold", type=float, default=0.08)
    parser.add_argument("--guard", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--sol-min-tokens", type=int, default=12288)
    parser.add_argument("--cache-device", choices=["cpu", "gpu"], default="cpu")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--allow-experimental-sol", action="store_true", help="Explicitly opt into the unvalidated Sol path")
    args = parser.parse_args()
    if args.size > 1024:
        parser.error("Large-image stress tests are suspended after a system restart; use 512 or 1024.")
    if "sol" in args.modes and not args.allow_experimental_sol:
        parser.error("Sol requires --allow-experimental-sol; full-model validation is incomplete.")
    lock = ROOT / "benchmark_results" / "run.lock"
    lock.parent.mkdir(exist_ok=True)
    with lock.open("x", encoding="utf-8") as handle:
        handle.write("Native API benchmark running. Do not run other GPU or canvas tests.\n")
    state = {"pending": False}
    try:
        asyncio.run(run(args, state))
    finally:
        if state["pending"]:
            print(f"Test may still be running. Lock retained: {lock}. Verify the server is idle before removing it.", flush=True)
        else:
            lock.unlink()
