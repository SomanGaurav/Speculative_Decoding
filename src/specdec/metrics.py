"""Metrics collection. AirLLM forward-call count is the project's primary success
metric (see README "Key architectural constraint") — everything else is secondary
and explicitly labeled as such wherever it's reported.
"""

from dataclasses import dataclass, field


@dataclass
class CallCounter:
    """Counts calls into AirLLM's forward(). This is the primary efficiency metric:
    baseline pays exactly 1 call per generated token; speculative decoding pays 1
    call per verification batch (ideally >1 accepted token per call)."""

    count: int = 0

    def increment(self) -> None:
        self.count += 1


@dataclass
class RunMetrics:
    airllm_calls: int = 0                  # primary metric
    wall_clock_seconds: float = 0.0        # secondary, caveated (AirLLM-dominated)
    generated_tokens: int = 0
    accepted_tokens: int = 0               # speculative only; == generated_tokens for baseline
    peak_vram_bytes: int = 0
    peak_ram_bytes: int = 0
    per_step_entropies: list[float] = field(default_factory=list)

    @property
    def tokens_per_call(self) -> float:
        return self.generated_tokens / self.airllm_calls if self.airllm_calls else 0.0

    @property
    def tokens_per_second(self) -> float:
        return self.generated_tokens / self.wall_clock_seconds if self.wall_clock_seconds else 0.0

    def to_dict(self) -> dict:
        return {
            "airllm_calls": self.airllm_calls,
            "wall_clock_seconds": self.wall_clock_seconds,
            "generated_tokens": self.generated_tokens,
            "accepted_tokens": self.accepted_tokens,
            "tokens_per_call": self.tokens_per_call,
            "tokens_per_second": self.tokens_per_second,
            "peak_vram_bytes": self.peak_vram_bytes,
            "peak_ram_bytes": self.peak_ram_bytes,
            "mean_entropy": (
                sum(self.per_step_entropies) / len(self.per_step_entropies)
                if self.per_step_entropies else None
            ),
        }
