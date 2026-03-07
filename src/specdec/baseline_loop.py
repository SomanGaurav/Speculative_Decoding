"""Baseline: plain AirLLM-only decoding, one token at a time. By construction this
pays exactly 1 AirLLM call per generated token — the reference point the speculative
pipeline's call-count reduction is measured against.
"""

import time

import psutil
import torch

from .metrics import CallCounter, RunMetrics
from .target_model import prefill as target_prefill
from .target_model import verify as target_verify


def generate_baseline(
    prompt_ids: torch.Tensor,
    target_model,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> tuple[torch.Tensor, RunMetrics]:
    t_start = time.time()
    torch.cuda.reset_peak_memory_stats()
    call_counter = CallCounter()
    metrics = RunMetrics()

    prompt_len = prompt_ids.shape[1]
    prefill_result = target_prefill(target_model, prompt_ids, call_counter)
    cache = prefill_result.cache
    next_token = prefill_result.probs.argmax(dim=-1, keepdim=True)

    generated_tokens = [next_token]
    position = prompt_len

    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
        result = target_verify(target_model, next_token, position, cache, call_counter)
        cache = result.cache
        next_token = result.probs.argmax(dim=-1, keepdim=True)
        generated_tokens.append(next_token)
        position += 1

    generated = torch.cat(generated_tokens, dim=1)

    metrics.airllm_calls = call_counter.count
    metrics.generated_tokens = generated.shape[1]
    metrics.accepted_tokens = generated.shape[1]  # every token is "accepted" by construction
    metrics.wall_clock_seconds = time.time() - t_start
    metrics.peak_vram_bytes = torch.cuda.max_memory_allocated()
    metrics.peak_ram_bytes = psutil.Process().memory_info().rss
    return generated, metrics
