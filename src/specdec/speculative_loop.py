"""Speculative decoding orchestration loop.

Cache invariant that makes each round cost exactly 1 AirLLM call for up to K+1
new tokens: after each round, `target_cache` covers every committed position
*except* the very last one (`last_committed_token` is held back, not yet fed
through the target model). The next round's verify() call always begins with
`last_committed_token` prepended to the new draft tokens — this simultaneously
(a) finishes committing that token's KV into the cache and (b) produces, for
free, the target's prediction for evaluating the very first draft token
(there is no separate "priming" call). See README for why this shape matters:
every AirLLM call pays a full disk sweep regardless of token count, so batching
K+1 tokens into 1 call is the entire point.

The draft model is *not* cached incrementally across rounds — it's fully
re-prefixed on the whole accepted sequence each round instead. This is a
deliberate simplification: the draft model runs entirely in VRAM and is fast
even without incremental caching, unlike AirLLM, where incremental caching is
the only thing that makes multi-token verification worthwhile at all.
"""

import time
from dataclasses import dataclass

import psutil
import torch

from .adaptive_length import LengthConfig
from .draft_model import draft_step
from .kv_cache_utils import crop_cache
from .metrics import CallCounter, RunMetrics
from .rejection_sampling import speculative_rejection_sample
from .target_model import prefill as target_prefill
from .target_model import verify as target_verify


@dataclass
class SpeculativeState:
    accepted_ids: torch.Tensor   # (1, accepted_len) full committed sequence so far
    target_cache: object          # covers accepted_len - 1 positions (last token held back)
    last_committed_token: torch.Tensor  # (1, 1)


def _draft_round(draft_model, accepted_ids: torch.Tensor, length_config: LengthConfig):
    """Full re-prefill of the draft model on the current accepted sequence, then
    continue autoregressively. Returns (draft_tokens (K,), draft_probs (K, vocab),
    entropies (list[float], length K))."""
    with torch.no_grad():
        out = draft_model(input_ids=accepted_ids, use_cache=True)
    cache = out.past_key_values
    first_probs = torch.softmax(out.logits[:, -1, :], dim=-1)
    first_token = first_probs.argmax(dim=-1, keepdim=True)
    first_entropy = -(first_probs * torch.log(first_probs.clamp_min(1e-12))).sum().item()

    continuation = draft_step(draft_model, first_token, cache, length_config)

    tokens = torch.cat([first_token.squeeze(0), continuation.tokens], dim=0)
    probs = torch.cat([first_probs, continuation.probs], dim=0)
    entropies = [first_entropy] + continuation.entropies
    return tokens, probs, entropies


def generate_speculative(
    prompt_ids: torch.Tensor,
    draft_model,
    target_model,
    max_new_tokens: int,
    length_config: LengthConfig,
    eos_token_id: int | None = None,
) -> tuple[torch.Tensor, RunMetrics]:
    t_start = time.time()
    torch.cuda.reset_peak_memory_stats()

    call_counter = CallCounter()
    metrics = RunMetrics()

    prompt_len = prompt_ids.shape[1]
    prefill_result = target_prefill(target_model, prompt_ids[:, :-1], call_counter)
    state = SpeculativeState(
        accepted_ids=prompt_ids,
        target_cache=prefill_result.cache,
        last_committed_token=prompt_ids[:, -1:],
    )

    generated = 0
    while generated < max_new_tokens:
        draft_tokens, draft_probs, entropies = _draft_round(
            draft_model, state.accepted_ids, length_config)
        metrics.per_step_entropies.extend(entropies)
        K = draft_tokens.shape[0]

        verify_input = torch.cat([state.last_committed_token, draft_tokens.unsqueeze(0)], dim=1)
        accepted_len = state.accepted_ids.shape[1]
        verify_result = target_verify(
            target_model, verify_input, accepted_len - 1, state.target_cache, call_counter,
        )

        target_probs_full = verify_result.probs  # (K+1, vocab): p_1..p_{K+1}
        # Draft and target models pad their lm_head output to different sizes for hardware
        # efficiency (e.g. Qwen2.5-3B: 151936, Qwen2.5-7B: 152064) even though they share the
        # same underlying tokenizer (151643 real tokens). Truncate both to the smaller padded
        # size before comparing per-token probabilities -- the truncated rows are unused
        # reserved ids with ~0 probability under a trained model, so this loses nothing real.
        shared_vocab = min(draft_probs.shape[-1], target_probs_full.shape[-1])
        result = speculative_rejection_sample(
            draft_tokens, draft_probs[:, :shared_vocab], target_probs_full[:, :shared_vocab],
        )

        new_tokens = torch.cat([result.accepted_tokens, result.bonus_token], dim=0)
        state.accepted_ids = torch.cat([state.accepted_ids, new_tokens.unsqueeze(0)], dim=1)

        crop_len = accepted_len + result.num_accepted
        crop_cache(verify_result.cache, crop_len)
        state.target_cache = verify_result.cache
        state.last_committed_token = result.bonus_token.unsqueeze(0)

        metrics.accepted_tokens += result.num_accepted
        generated += new_tokens.shape[0]

        if eos_token_id is not None and (new_tokens == eos_token_id).any():
            break

    metrics.airllm_calls = call_counter.count
    metrics.generated_tokens = generated
    metrics.wall_clock_seconds = time.time() - t_start
    metrics.peak_vram_bytes = torch.cuda.max_memory_allocated()
    metrics.peak_ram_bytes = psutil.Process().memory_info().rss
    return state.accepted_ids[:, prompt_len:], metrics
