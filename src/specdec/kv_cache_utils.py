"""KV-cache helpers for the draft/verify speculative decoding loop.

Confirmed by scripts/00_spike_airllm_forward.py (see README "Implementation Risk
Findings"): AirLLM's forward() supports true incremental KV-cache reuse across the
draft/verify boundary, including multi-token verification calls and post-rejection
cache rollback via DynamicCache.crop(). Two things are required for this to work
correctly, both handled here:

  1. `cache_position` must always be passed explicitly. Without it, transformers
     defaults to range(0, seq_len) instead of range(past_len, past_len+seq_len),
     silently corrupting position embeddings/attention for any call that carries an
     existing past_key_values.
  2. DynamicCache.update() mutates the cache object in place, so any cache that needs
     to be reused for more than one downstream call must be cloned first
     (torch.Tensor.__deepcopy__ refuses non-leaf tensors, hence the manual clone).
"""

import torch
from transformers.cache_utils import DynamicCache


def clone_cache(cache: DynamicCache) -> DynamicCache:
    """Copy a DynamicCache's key/value tensors into a fresh cache object.

    Needed anywhere a cache is about to be passed into a forward call but must
    still be usable afterward (forward calls mutate their input cache in place).
    """
    new_cache = DynamicCache()
    for idx, layer in enumerate(cache.layers):
        new_cache.update(layer.keys.clone(), layer.values.clone(), idx)
    return new_cache


def cache_position_for(past_length: int, num_new_tokens: int, device: torch.device) -> torch.Tensor:
    """Position ids for `num_new_tokens` new positions following `past_length`
    already-cached positions. Must be passed to every forward() call that supplies
    a non-empty past_key_values."""
    return torch.arange(past_length, past_length + num_new_tokens, device=device)


def crop_cache(cache: DynamicCache, new_length: int) -> DynamicCache:
    """Truncate a cache back to `new_length` positions in place (post-rejection rollback)."""
    cache.crop(new_length)
    return cache


def build_verify_input(
    prompt_len: int,
    accepted_len: int,
    new_token_ids: torch.Tensor,
    cache: DynamicCache,
) -> tuple[torch.Tensor, torch.Tensor, DynamicCache]:
    """Build the (input_ids, cache_position, cache) triple for a verification forward
    call that checks `new_token_ids` against a cache already covering
    `prompt_len + accepted_len` positions.

    Returns a cloned cache so the caller's original cache is left untouched (the
    orchestration loop needs the pre-verification cache preserved for rollback).
    """
    past_length = prompt_len + accepted_len
    device = new_token_ids.device
    cache_position = cache_position_for(past_length, new_token_ids.shape[-1], device)
    return new_token_ids, cache_position, clone_cache(cache)
