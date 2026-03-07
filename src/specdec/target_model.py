"""Target model verification wrapper (AirLLM, disk-streamed Qwen2.5-7B).

verify() is exactly one AirLLM forward() call per invocation — the project's primary
efficiency metric (see metrics.CallCounter) counts these directly. Every call pays a
full disk sweep through the model's layers regardless of how many draft tokens are
being verified (see README "Key architectural constraint"), which is exactly why
batching K draft tokens into one verify() call is the point: fewer, larger calls
instead of one call per token.
"""

from dataclasses import dataclass

import torch
from airllm import AutoModel

from .kv_cache_utils import cache_position_for, clone_cache
from .metrics import CallCounter


@dataclass
class VerifyResult:
    probs: torch.Tensor   # (num_new_tokens, vocab) target model's per-position probabilities
    cache: object          # resulting DynamicCache, covering prompt + all verified positions


def load_target_model(model_id: str, device: str = "cuda:0", compression: str | None = "4bit",
                       max_seq_len: int = 2048):
    return AutoModel.from_pretrained(model_id, device=device, compression=compression,
                                      max_seq_len=max_seq_len)


@torch.no_grad()
def prefill(model, prompt_ids: torch.Tensor, call_counter: CallCounter) -> VerifyResult:
    """Initial prefill call on the prompt. Unavoidable — 1 AirLLM call, same for both
    the baseline and speculative pipelines."""
    out = model(input_ids=prompt_ids, use_cache=True)
    call_counter.increment()
    probs = torch.softmax(out.logits[:, -1, :], dim=-1)
    return VerifyResult(probs=probs, cache=out.past_key_values)


@torch.no_grad()
def verify(
    model,
    new_token_ids: torch.Tensor,   # (1, K) candidate tokens to check, K>=1
    prompt_len_plus_accepted: int,  # position count already covered by `cache`
    cache,
    call_counter: CallCounter,
) -> VerifyResult:
    """One AirLLM forward call verifying all of `new_token_ids` at once against `cache`.
    Returns per-position target probabilities (used for rejection sampling) and the
    resulting cache (cloned from the input, so the caller's cache is untouched and can
    still be used for rollback bookkeeping by the orchestration loop)."""
    device = new_token_ids.device
    working_cache = clone_cache(cache)
    cache_position = cache_position_for(prompt_len_plus_accepted, new_token_ids.shape[-1], device)

    out = model(input_ids=new_token_ids, past_key_values=working_cache, use_cache=True,
                cache_position=cache_position)
    call_counter.increment()

    probs = torch.softmax(out.logits[0], dim=-1)
    return VerifyResult(probs=probs, cache=out.past_key_values)


@torch.no_grad()
def decode_one_token(
    model,
    token_id: torch.Tensor,   # (1, 1)
    position: int,
    cache,
    call_counter: CallCounter,
) -> VerifyResult:
    """Baseline path: verify() specialized to K=1, i.e. one AirLLM call per output token."""
    return verify(model, token_id, position, cache, call_counter)
