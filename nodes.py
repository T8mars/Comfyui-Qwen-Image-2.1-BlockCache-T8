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


def window(start, end, threshold_mode="constant", split_ratio=0.5):
    if not 0 <= start < end <= 1:
        raise ValueError("start_percent must be less than end_percent, within [0, 1]")
    if threshold_mode not in ("constant", "two_stage"):
        raise ValueError("threshold_mode must be constant or two_stage")
    if threshold_mode == "two_stage" and not 0 <= split_ratio <= 1:
        raise ValueError("split_ratio must be within [0, 1]")


class QwenImage21BlockCacheT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImage21BlockCacheT8", display_name="Qwen Image 2.1 Block Cache (T8)", category=CATEGORY,
            description="First-block target residual cache, preserving Core's reference/text KV cache. For editing, start with threshold 0.03; higher values may change pose and details.",
            is_experimental=True,
            inputs=[io.Model.Input("model"),
                    io.Float.Input("residual_diff_threshold", default=0.08, min=0, max=1, step=0.01, tooltip="Fixed threshold, or early-stage threshold in two_stage mode. 固定阈值 / 两段模式的前段阈值。"),
                    io.Float.Input("start_percent", default=0.10, min=0, max=1, step=0.001, round=0.001, tooltip="Native sampling progress, not a step count. Before this progress, no Block reuse. 开始进度。"),
                    io.Float.Input("end_percent", default=0.85, min=0, max=1, step=0.001, round=0.001, tooltip="Native sampling progress, not a step count. From this progress onward, no Block reuse. 结束进度。"),
                    io.Int.Input("max_consecutive_hits", default=2, min=1, max=10),
                    io.Combo.Input("cache_device", options=["cpu", "gpu"], default="cpu"),
                    io.Int.Input("metric_stride", default=8, min=1, max=32),
                    io.Int.Input("max_cache_mb", default=1024, min=16, max=32768),
                    io.Combo.Input("threshold_mode", options=["constant", "two_stage"], default="constant", optional=True, tooltip="constant preserves old workflows. two_stage splits the active progress window into early/late thresholds. 两段阈值模式，按进度而非步数。"),
                    io.Float.Input("split_ratio", default=0.5, min=0, max=1, step=0.01, optional=True, tooltip="Early share of the active window, not the whole sampling run. 0.5 = half early, half late. 可跳层区间内的前段比例。"),
                    io.Float.Input("late_threshold", default=0.03, min=0, max=1, step=0.01, optional=True, tooltip="Late-stage residual threshold, used only in two_stage mode. 后段阈值，0表示该节点在后段不跳层。")],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, residual_diff_threshold=0.08, start_percent=0.10, end_percent=0.85,
                max_consecutive_hits=2, cache_device="cpu", metric_stride=8, max_cache_mb=1024,
                threshold_mode="constant", split_ratio=0.5, late_threshold=0.03):
        require_qwen21(model)
        window(start_percent, end_percent, threshold_mode, split_ratio)
        config = CacheConfig(residual_diff_threshold, start_percent, end_percent, max_consecutive_hits, cache_device, metric_stride, max_cache_mb,
                             threshold_mode, split_ratio, late_threshold)
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
                    io.Float.Input("guard_threshold", default=0.25, min=0, max=1, step=0.01, tooltip="Fixed guard threshold, or early-stage guard in two_stage mode. 固定阈值 / 两段模式的前段阈值。"),
                    io.Float.Input("start_percent", default=0.15, min=0, max=1, step=0.001, round=0.001, tooltip="Native sampling progress, not a step count. Before this progress, no Spectrum prediction. 开始进度。"),
                    io.Float.Input("end_percent", default=0.85, min=0, max=1, step=0.001, round=0.001, tooltip="Native sampling progress, not a step count. From this progress onward, no Spectrum prediction. 结束进度。"),
                    io.Int.Input("max_consecutive_hits", default=1, min=1, max=10),
                    io.Combo.Input("cache_device", options=["cpu", "gpu"], default="cpu"),
                    io.Int.Input("max_cache_mb", default=1024, min=16, max=32768),
                    io.Combo.Input("threshold_mode", options=["constant", "two_stage"], default="constant", optional=True, tooltip="constant preserves old workflows. two_stage splits the active progress window into early/late thresholds. 两段阈值模式，按进度而非步数。"),
                    io.Float.Input("split_ratio", default=0.5, min=0, max=1, step=0.01, optional=True, tooltip="Early share of the active window, not the whole sampling run. 0.5 = half early, half late. 可跳层区间内的前段比例。"),
                    io.Float.Input("late_threshold", default=0.08, min=0, max=1, step=0.01, optional=True, tooltip="Late-stage guard threshold, used only in two_stage mode. 后段阈值，0表示该节点在后段不预测。")],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, history=4, degree=2, ridge=0.01, guard_threshold=0.25, start_percent=0.15,
                end_percent=0.85, max_consecutive_hits=1, cache_device="cpu", max_cache_mb=1024,
                threshold_mode="constant", split_ratio=0.5, late_threshold=0.08):
        require_qwen21(model)
        window(start_percent, end_percent, threshold_mode, split_ratio)
        if history <= degree or ridge <= 0:
            raise ValueError("Spectrum history must exceed degree, and ridge must be positive")
        config = SpectrumConfig(history, degree, ridge, guard_threshold, start_percent, end_percent, max_consecutive_hits, cache_device, max_cache_mb,
                                threshold_mode, split_ratio, late_threshold)
        return io.NodeOutput(install(model, "spectrum", config))


class QwenImage21SageAttentionT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImage21SageAttentionT8", display_name="Qwen Image 2.1 Sage Attention (T8)", category=CATEGORY,
            description="Model-local Sage. Optional sage_kitchen routes long unmasked D128 attention to Kitchen, preserving Sage's native path for short/masked calls. Both use Core adapters; speed depends on GPU and shape.",
            inputs=[io.Model.Input("model"),
                    io.Combo.Input("backend_mode", options=["sage", "sage_kitchen"], default="sage", optional=True,
                                   tooltip="sage preserves old behavior. sage_kitchen mixes dense backends by shape; no skipped layers or runtime tuning. 混合模式无需再串接 Kitchen 选择节点。")],
            outputs=[io.Model.Output()], is_experimental=True,
        )

    @classmethod
    def execute(cls, model, backend_mode="sage"):
        require_qwen21(model)
        if backend_mode not in ("sage", "sage_kitchen"):
            raise ValueError("backend_mode must be sage or sage_kitchen")
        if not native_attention.SAGE_ATTENTION_IS_AVAILABLE:
            raise RuntimeError("SageAttention is not installed in this ComfyUI Python environment")
        return io.NodeOutput(install(model, "sage", True if backend_mode == "sage" else backend_mode))


class QwenImage21SolAttentionT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="QwenImage21SolAttentionT8", display_name="Qwen Image 2.1 Sol Attention (T8)", category=CATEGORY,
            description="Opt-in experimental Sol adapter. Limited 1024 testing passed with Core compilation disabled; image details changed. A 2048 test restarted the system; cause unresolved. Disabled by default.",
            inputs=[io.Model.Input("model"),
                    io.Float.Input("tau", default=1.0, min=0, max=4, step=0.05),
                    io.Int.Input("min_tokens", default=12288, min=64, max=131072, step=64),
                    io.Float.Input("start_percent", default=0.15, min=0, max=1, step=0.01),
                    io.Float.Input("end_percent", default=0.85, min=0, max=1, step=0.01),
                    io.Boolean.Input("enabled", default=False, tooltip="Opt in only for comparison tests. Limited 1024 validation; 2048 stability unresolved.")],
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
