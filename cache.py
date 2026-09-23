from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import math

import torch


@dataclass(frozen=True)
class CacheConfig:
    threshold: float = 0.08
    start_percent: float = 0.10
    end_percent: float = 0.85
    max_consecutive_hits: int = 2
    cache_device: str = "cpu"
    metric_stride: int = 8
    max_cache_mb: int = 1024


@dataclass(frozen=True)
class SpectrumConfig:
    history: int = 4
    degree: int = 2
    ridge: float = 0.01
    guard_threshold: float = 0.25
    start_percent: float = 0.15
    end_percent: float = 0.85
    max_consecutive_hits: int = 1
    cache_device: str = "cpu"
    max_cache_mb: int = 1024


@dataclass
class Stream:
    last_sigma: float | None = None
    indicator: torch.Tensor | None = None
    history: list = field(default_factory=list)
    consecutive: int = 0

    def clear(self):
        self.indicator = None
        self.history.clear()
        self.consecutive = 0


def chebyshev_row(x, degree):
    row = [1.0]
    if degree:
        row.append(x)
    for _ in range(2, degree + 1):
        row.append(2 * x * row[-1] - row[-2])
    return row


def forecast_weights(sigmas, sigma, degree, ridge):
    """Small CPU solve; never build a feature-sized regression coefficient matrix."""
    span = max(sigmas) - min(sigmas)
    if span <= 1e-8:
        return None
    origin = min(sigmas)
    design = torch.tensor([chebyshev_row(2 * (s - origin) / span - 1, degree) for s in sigmas], dtype=torch.float64)
    query = torch.tensor(chebyshev_row(2 * (sigma - origin) / span - 1, degree), dtype=torch.float64)
    penalty = torch.eye(degree + 1, dtype=torch.float64) * ridge
    penalty[0, 0] = 0
    weights = design @ torch.linalg.solve(design.T @ design + penalty, query)
    # A normal one-step quadratic forecast over four anchors has L1 norm ~5.
    if not torch.isfinite(weights).all() or weights.abs().sum() > 8:
        return None
    return weights.tolist()


class CacheRuntime:
    def __init__(self, block, spectrum, model_sampling):
        self.block = block
        self.spectrum = spectrum
        configs = [c for c in (block, spectrum) if c is not None]
        self.windows = {id(c): (float(model_sampling.percent_to_sigma(c.start_percent)), float(model_sampling.percent_to_sigma(c.end_percent))) for c in configs}
        self.device = "cpu" if any(c.cache_device == "cpu" for c in configs) else "gpu"
        self.budget = min(c.max_cache_mb for c in configs) * 1024 * 1024
        self.stride = block.metric_stride if block else 8
        self.streams = OrderedDict()
        self.full = 0
        self.hits = 0
        self.forecasts = 0
        self.peak_bytes = 0

    def copy(self, value):
        return value.detach().to(device="cpu" if self.device == "cpu" else value.device, copy=True).contiguous()

    def sample(self, value):
        return self.copy(value[:, ::self.stride, ::self.stride].float())

    def window(self, config, sigma):
        start, end = self.windows[id(config)]
        return end < sigma <= start

    def stream(self, key, sigma):
        stream = self.streams.pop(key, Stream())
        if stream.last_sigma is not None and sigma >= stream.last_sigma:
            stream.clear()
        stream.last_sigma = sigma
        self.streams[key] = stream
        return stream

    @staticmethod
    def difference(current, previous):
        # A single unstable CFG/batch row must force a real forward.
        axes = tuple(range(1, current.ndim))
        return float(((current - previous).abs().mean(axes) / previous.abs().mean(axes).clamp_min(1e-6)).max())

    def replay(self, stream, indicator, sigma, target):
        if not stream.history or stream.indicator is None:
            return None, None
        score = self.difference(indicator, stream.indicator)
        if not math.isfinite(score):
            return None, None
        if self.block and stream.consecutive < self.block.max_consecutive_hits and self.window(self.block, sigma) and score < self.block.threshold:
            result = target + stream.history[-1][1].to(target)
            if not torch.isfinite(result).all():
                return None, None
            self.hits += 1
            stream.consecutive += 1
            return result, "cache"
        config = self.spectrum
        if config and stream.consecutive < config.max_consecutive_hits and self.window(config, sigma) and score < config.guard_threshold and len(stream.history) >= config.history:
            weights = forecast_weights([s for s, _ in stream.history], sigma, config.degree, config.ridge)
            if weights is not None:
                result = target.clone()
                # Bound fp32 scratch independently of resolution/history size.
                for start in range(0, target.shape[1], 512):
                    end = start + 512
                    predicted = torch.zeros_like(stream.history[-1][1][:, start:end], dtype=torch.float32)
                    for weight, (_, tail) in zip(weights, stream.history):
                        predicted.add_(tail[:, start:end], alpha=weight)
                    if not torch.isfinite(predicted).all():
                        return None, None
                    result[:, start:end].add_(predicted.to(target))
                if not torch.isfinite(result).all():
                    return None, None
                self.forecasts += 1
                stream.consecutive += 1
                return result, "spectrum"
        return None, None

    def store(self, stream, indicator, sigma, target, anchor):
        tail = self.copy(target - anchor)
        if not torch.isfinite(tail).all():
            stream.clear()
            return
        stream.history.append((sigma, tail))
        keep = self.spectrum.history if self.spectrum else 1
        del stream.history[:-keep]
        stream.indicator = indicator
        stream.consecutive = 0
        self.trim()

    def bytes(self):
        return sum(t.numel() * t.element_size() for s in self.streams.values() for t in ([s.indicator] if s.indicator is not None else []) + [v for _, v in s.history])

    def trim(self):
        while self.bytes() > self.budget and self.streams:
            self.streams.popitem(last=False)
        self.peak_bytes = max(self.peak_bytes, self.bytes())

    def clear(self):
        self.streams.clear()


@dataclass
class ForwardState:
    runtime: CacheRuntime
    stream: Stream
    sigma: float
    target_tokens: int
    active: bool
    indicator: torch.Tensor | None = None
    anchor: torch.Tensor | None = None


class CacheHit(Exception):
    def __init__(self, target):
        super().__init__()
        self.target = target
