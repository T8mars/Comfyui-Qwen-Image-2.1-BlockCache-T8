"""Opt-in, tiny CUDA kernel probe. No weights, text encoder, VAE, or server."""
# ruff: noqa: T201
import importlib.util
from pathlib import Path
import types

import comfy_kitchen as ck
import torch


def main():
    path = Path(__file__).resolve().parents[1] / "attention.py"
    spec = importlib.util.spec_from_file_location("qwen_attention_probe", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not ck.sol_attn_is_available(torch.device("cuda")):
        raise RuntimeError("Compiled Sol kernel unavailable on this GPU")
    torch.manual_seed(42)
    for dtype in (torch.bfloat16, torch.float16):
        q = torch.randn(1, 288, 2, 128, dtype=dtype, device="cuda")
        k = torch.randn(1, 321, 2, 128, dtype=dtype, device="cuda")
        v = torch.randn_like(k)
        actual = module.square_target_attention(q, k, v, 0.8)
        with ck.use_backend("eager"):
            expected = module.square_target_attention(q.float(), k.float(), v.float(), 0.8)
        relative = float((actual.float() - expected).norm() / expected.norm())
        if not torch.isfinite(actual).all() or relative > 0.08:
            raise AssertionError(f"Compiled/eager Sol mismatch: {dtype}: {relative}")
        print(f"Sol {dtype}: compiled/eager relative L2={relative:.6f}, shape={tuple(actual.shape)}")
    from comfy.ldm.modules import attention as native
    # Exercise actual registered dense backends on rectangular image queries and causal text masks.
    for backend in (native.attention_sage, native.attention_comfy_kitchen_int8):
        for masked in (False, True):
            mask = torch.ones(288, 321, device="cuda", dtype=torch.bool).tril(33) if masked else None
            out = backend(q.flatten(2), k.flatten(2), v.flatten(2), 2, mask=mask)
            assert out.shape == (1, 288, 256) and torch.isfinite(out).all()
            print(f"dense backend masked={masked}: finite shape={tuple(out.shape)}")
    stats = {}
    model = types.SimpleNamespace(transformer_blocks=[])
    override = module.make_override(model, None, True, module.SolConfig(min_tokens=64), 288, 321, 0.5, stats)
    output = override(native.attention_pytorch, q.flatten(2), k.flatten(2), v.flatten(2), 2)
    assert output.shape == (1, 288, 256) and stats.get("sol") == 1
    print(f"T8 composed override stats={stats}")
    torch.cuda.synchronize()
    print(f"peak allocated tensor memory={torch.cuda.max_memory_allocated() / 1048576:.2f} MiB")


if __name__ == "__main__":
    lock = Path(__file__).resolve().parents[1] / "benchmark_results" / "run.lock"
    lock.parent.mkdir(exist_ok=True)
    with lock.open("x", encoding="utf-8") as handle:
        handle.write("Small GPU probe running. Do not run other GPU or canvas tests.\n")
    try:
        main()
    finally:
        lock.unlink()
