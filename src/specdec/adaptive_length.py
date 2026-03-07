"""Adaptive speculation length: how many tokens the draft model proposes per step.

Two modes:
  - entropy_threshold: keep drafting while the draft model's own per-token entropy
    stays below a threshold (confident draft -> speculate further), up to max_k.
  - fixed_k: always draft exactly max_k tokens, no entropy check. Used as a control
    condition to measure whether the adaptive heuristic actually helps.

Deliberately not a trained controller (no CM-ASD/SVIP/EAGLE-2 reimplementation) --
a transparent threshold is enough to compare adaptive vs. fixed-length speculation.
"""

from dataclasses import dataclass


@dataclass
class LengthConfig:
    mode: str = "entropy_threshold"   # "entropy_threshold" | "fixed_k"
    max_k: int = 6
    entropy_threshold: float = 2.0    # nats; lower = stricter (shorter lookahead)


def should_continue_drafting(step_index: int, entropy: float, config: LengthConfig) -> bool:
    """Called after drafting `step_index + 1` tokens (0-indexed) with the latest token's
    entropy. Returns whether the draft model should propose another token."""
    if step_index + 1 >= config.max_k:
        return False
    if config.mode == "fixed_k":
        return True
    if config.mode == "entropy_threshold":
        return entropy < config.entropy_threshold
    raise ValueError(f"unknown adaptive length mode: {config.mode!r}")
