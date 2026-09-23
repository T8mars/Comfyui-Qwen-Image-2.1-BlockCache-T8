from dataclasses import dataclass

import torch
import comfy_kitchen as ck
from comfy.ldm.modules import attention as native_attention


@dataclass(frozen=True)
class SolConfig:
    tau: float = 1.0
    min_tokens: int = 12288
    start_percent: float = 0.15
    end_percent: float = 0.85


def square_target_attention(q, k, v, tau, scale=None):
    """BTHD target queries -> square Sol kernel; prefix KV remains exact."""
    prefix = k.shape[1] - q.shape[1]
    padded = q.new_zeros(k.shape) if prefix else q
    if prefix:
        padded[:, prefix:] = q
    # Mixed query blocks must not be routed using artificial prefix queries.
    boundary = [prefix // 64, (prefix + 63) // 64] if prefix % 64 else [0, 0]
    result = ck.sol_attn(
        padded.contiguous(), k.contiguous(), v.contiguous(), tau=tau, scale=scale,
        sink_blocks=[0, (prefix + 63) // 64], sink_q=boundary,
    )
    return result[:, prefix:]


def make_override(model, previous, sage, sol, target_tokens, total_tokens, progress, stats):
    def override(func, q, k, v, heads, mask=None, attn_precision=None,
                 skip_reshape=False, skip_output_reshape=False, **kwargs):
        common = dict(mask=mask, attn_precision=attn_precision, skip_reshape=skip_reshape,
                      skip_output_reshape=skip_output_reshape, **kwargs)
        # Calling native dispatchers from an override must not enter this override again.
        common["_inside_attn_wrapper"] = True

        def dense():
            if sage and q.device.type == "cuda" and q.dtype in (torch.float16, torch.bfloat16) and not any(t.requires_grad for t in (q, k, v)):
                stats["sage"] = stats.get("sage", 0) + 1
                return native_attention.attention_sage(q, k, v, heads, **common)
            if previous is not None:
                return previous(func, q, k, v, heads, **common)
            index = kwargs.get("transformer_options", {}).get("block_index", 0)
            preferred = model.transformer_blocks[index].attn.comfy_attention.function
            return (preferred if preferred is not None else func)(q, k, v, heads, **common)

        if sol is None:
            return dense()
        q_len = q.shape[2] if skip_reshape else q.shape[1]
        k_len = k.shape[2] if skip_reshape else k.shape[1]
        dim = q.shape[-1] if skip_reshape else q.shape[-1] // heads
        eligible = (
            mask is None and kwargs.get("low_precision_attention", True)
            and not any(t.requires_grad for t in (q, k, v))
            and q.device.type == "cuda" and k.device == q.device and v.device == q.device
            and q.dtype in (torch.float16, torch.bfloat16) and q.dtype == k.dtype == v.dtype
            and dim == 128 and q_len == target_tokens and k_len == total_tokens
            and k.shape == v.shape and target_tokens >= sol.min_tokens
            and progress is not None and sol.start_percent <= progress <= sol.end_percent
        )
        if not eligible or not ck.sol_attn_is_available(q.device):
            stats["sol_dense"] = stats.get("sol_dense", 0) + 1
            return dense()
        if skip_reshape:
            qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
        else:
            qs, ks, vs = (t.reshape(t.shape[0], -1, heads, dim) for t in (q, k, v))
        result = square_target_attention(qs, ks, vs, sol.tau, kwargs.get("scale"))
        stats["sol"] = stats.get("sol", 0) + 1
        return result.transpose(1, 2) if skip_output_reshape else result.reshape(q.shape[0], q_len, heads * dim)

    return override
