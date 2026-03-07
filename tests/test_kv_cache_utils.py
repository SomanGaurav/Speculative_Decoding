"""KV-cache utils, validated against a tiny CPU-only HF model (no AirLLM, no GPU) so
cache-manipulation logic gets fast, repeatable feedback independent of AirLLM's
disk-streaming latency.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from specdec.kv_cache_utils import build_verify_input, cache_position_for, clone_cache, crop_cache

MODEL_ID = "sshleifer/tiny-gpt2"


@pytest.fixture(scope="module")
def model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.eval()
    return model, tokenizer


def test_clone_cache_is_independent_of_original(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    ids = tokenizer("hello world", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    original = out.past_key_values
    original_len = original.get_seq_length()

    clone = clone_cache(original)
    # mutate the clone via a forward call, original must be untouched
    cp = cache_position_for(original_len, 1, ids.device)
    with torch.no_grad():
        model(input_ids=ids[:, :1], past_key_values=clone, use_cache=True, cache_position=cp)

    assert original.get_seq_length() == original_len
    assert clone.get_seq_length() == original_len + 1


def test_multi_token_verify_matches_full_reprefill(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt_ids = tokenizer("hello world", return_tensors="pt").input_ids
    with torch.no_grad():
        prefill_out = model(input_ids=prompt_ids, use_cache=True)
    cache = prefill_out.past_key_values
    past_len = prompt_ids.shape[1]

    new_tokens = torch.tensor([[7, 8, 9]])
    input_ids, cache_position, verify_cache = build_verify_input(past_len, 0, new_tokens, cache)

    with torch.no_grad():
        cached_out = model(input_ids=input_ids, past_key_values=verify_cache, use_cache=True,
                            cache_position=cache_position)
        full_ids = torch.cat([prompt_ids, new_tokens], dim=1)
        full_out = model(input_ids=full_ids, use_cache=True)

    cached_logits = cached_out.logits
    full_logits = full_out.logits[:, -new_tokens.shape[1]:, :]
    assert torch.equal(cached_logits.argmax(dim=-1), full_logits.argmax(dim=-1))

    # original cache passed to build_verify_input must be untouched
    assert cache.get_seq_length() == past_len


def test_build_verify_input_does_not_mutate_source_cache(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt_ids = tokenizer("hi", return_tensors="pt").input_ids
    with torch.no_grad():
        out = model(input_ids=prompt_ids, use_cache=True)
    cache = out.past_key_values
    len_before = cache.get_seq_length()

    build_verify_input(prompt_ids.shape[1], 0, torch.tensor([[1, 2]]), cache)
    assert cache.get_seq_length() == len_before


def test_crop_cache_truncates_and_matches_shorter_reprefill(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt_ids = tokenizer("hello world", return_tensors="pt").input_ids
    with torch.no_grad():
        prefill_out = model(input_ids=prompt_ids, use_cache=True)
    cache = prefill_out.past_key_values
    past_len = prompt_ids.shape[1]

    new_tokens = torch.tensor([[7, 8, 9]])
    input_ids, cache_position, verify_cache = build_verify_input(past_len, 0, new_tokens, cache)
    with torch.no_grad():
        model(input_ids=input_ids, past_key_values=verify_cache, use_cache=True,
              cache_position=cache_position)

    # simulate rejecting everything after the first new token: roll back to past_len + 1
    rollback_len = past_len + 1
    crop_cache(verify_cache, rollback_len)
    assert verify_cache.get_seq_length() == rollback_len

    probe = new_tokens[:, :1]
    cp = cache_position_for(rollback_len, 1, probe.device)
    with torch.no_grad():
        rolled_back_out = model(input_ids=probe, past_key_values=verify_cache, use_cache=True,
                                 cache_position=cp)
        reference_ids = torch.cat([prompt_ids, new_tokens[:, :1], probe], dim=1)
        reference_out = model(input_ids=reference_ids, use_cache=True)

    assert torch.equal(
        rolled_back_out.logits[:, -1, :].argmax(dim=-1),
        reference_out.logits[:, -1, :].argmax(dim=-1),
    )
