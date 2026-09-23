from comfy_api.latest import ComfyExtension, io
from comfy.ldm.qwen_image21.model import QwenImage21Transformer2DModel
from comfy.ldm.modules import attention as native_attention

from .attention import SolConfig
from .cache import CacheConfig, SpectrumConfig
from .runtime import install


CATEGORY = "T8/Qwen Image 2.1"


def require_qwen21(model):
    if not isinstance(model.model.diffusion_model, QwenImage21Transformer2DModel):
        raise ValueError("This T8 node requires ComfyUI's native Qwen-Image-2.1 model (not Qwen Image 1.x)")


def window(start, end):
    if not 0 <= start < end <= 1:
        raise ValueError("start_percent must be less than end_percent, within [0, 1]")


class QwenImage21BlockCacheT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImage21BlockCacheT8", display_name="Qwen Image 2.1 Block Cache (T8)", category=CATEGORY,
            description="Experimental first-block residual cache, target tokens only. Disables Core prefix KV caching; quality/speed require comparison.",
            is_experimental=True,
            inputs=[io.Model.Input("model"),
                    io.Float.Input("residual_diff_threshold", default=0.08, min=0, max=1, step=0.01),
                    io.Float.Input("start_percent", default=0.10, min=0, max=1, step=0.01),
                    io.Float.Input("end_percent", default=0.85, min=0, max=1, step=0.01),
                    io.Int.Input("max_consecutive_hits", default=2, min=1, max=10),
                    io.Combo.Input("cache_device", options=["cpu", "gpu"], default="cpu"),
                    io.Int.Input("metric_stride", default=8, min=1, max=32),
                    io.Int.Input("max_cache_mb", default=1024, min=16, max=32768)],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, residual_diff_threshold=0.08, start_percent=0.10, end_percent=0.85,
                max_consecutive_hits=2, cache_device="cpu", metric_stride=8, max_cache_mb=1024):
        require_qwen21(model)
        window(start_percent, end_percent)
        config = CacheConfig(residual_diff_threshold, start_percent, end_percent, max_consecutive_hits, cache_device, metric_stride, max_cache_mb)
        return io.NodeOutput(install(model, "block", config))


class QwenImage21SpectrumT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImage21SpectrumT8", display_name="Qwen Image 2.1 Spectrum (T8)", category=CATEGORY,
            description="Experimental Spectrum-inspired Chebyshev forecast of the target tail residual. Fits real forwards only; composes with T8 Block Cache.",
            is_experimental=True,
            inputs=[io.Model.Input("model"),
                    io.Int.Input("history", default=4, min=3, max=8),
                    io.Int.Input("degree", default=2, min=1, max=3),
                    io.Float.Input("ridge", default=0.01, min=0.0001, max=1, step=0.001),
                    io.Float.Input("guard_threshold", default=0.25, min=0, max=1, step=0.01),
                    io.Float.Input("start_percent", default=0.15, min=0, max=1, step=0.01),
                    io.Float.Input("end_percent", default=0.85, min=0, max=1, step=0.01),
                    io.Int.Input("max_consecutive_hits", default=1, min=1, max=10),
                    io.Combo.Input("cache_device", options=["cpu", "gpu"], default="cpu"),
                    io.Int.Input("max_cache_mb", default=1024, min=16, max=32768)],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, history=4, degree=2, ridge=0.01, guard_threshold=0.25, start_percent=0.15,
                end_percent=0.85, max_consecutive_hits=1, cache_device="cpu", max_cache_mb=1024):
        require_qwen21(model)
        window(start_percent, end_percent)
        if history <= degree or ridge <= 0:
            raise ValueError("Spectrum history must exceed degree, and ridge must be positive")
        config = SpectrumConfig(history, degree, ridge, guard_threshold, start_percent, end_percent, max_consecutive_hits, cache_device, max_cache_mb)
        return io.NodeOutput(install(model, "spectrum", config))


class QwenImage21SageAttentionT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImage21SageAttentionT8", display_name="Qwen Image 2.1 Sage Attention (T8)", category=CATEGORY,
            description="Model-local Sage via Core's masked/rectangular attention adapter. Requires an installed SageAttention backend.",
            inputs=[io.Model.Input("model")], outputs=[io.Model.Output()], is_experimental=True,
        )

    @classmethod
    def execute(cls, model):
        require_qwen21(model)
        if not native_attention.SAGE_ATTENTION_IS_AVAILABLE:
            raise RuntimeError("SageAttention is not installed in this ComfyUI Python environment")
        return io.NodeOutput(install(model, "sage", True))


class QwenImage21SolAttentionT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImage21SolAttentionT8", display_name="Qwen Image 2.1 Sol Attention (T8)", category=CATEGORY,
            description="Opt-in experimental Sol adapter. A 2048 full-model test ended in a system restart; cause unresolved. Disabled by default; no verified end-to-end speedup.",
            inputs=[io.Model.Input("model"),
                    io.Float.Input("tau", default=1.0, min=0, max=4, step=0.05),
                    io.Int.Input("min_tokens", default=12288, min=64, max=131072, step=64),
                    io.Float.Input("start_percent", default=0.15, min=0, max=1, step=0.01),
                    io.Float.Input("end_percent", default=0.85, min=0, max=1, step=0.01),
                    io.Boolean.Input("enabled", default=False, tooltip="Leave disabled unless explicitly testing this unvalidated sparse kernel path.")],
            outputs=[io.Model.Output()], is_experimental=True,
        )

    @classmethod
    def execute(cls, model, tau=1.0, min_tokens=12288, start_percent=0.15, end_percent=0.85, enabled=False):
        require_qwen21(model)
        window(start_percent, end_percent)
        return io.NodeOutput(install(model, "sol", SolConfig(tau, min_tokens, start_percent, end_percent) if enabled else None))


class QwenImage21T8Extension(ComfyExtension):
    async def get_node_list(self):
        return [QwenImage21BlockCacheT8, QwenImage21SpectrumT8, QwenImage21SageAttentionT8, QwenImage21SolAttentionT8]


def comfy_entrypoint():
    return QwenImage21T8Extension()
