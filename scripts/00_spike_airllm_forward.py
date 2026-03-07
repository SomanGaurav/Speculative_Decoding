"""Go/no-go spike: does AirLLM's forward() support multi-token verification + KV-cache reuse?

Tests, in order, against a small ungated Llama-architecture model (fast to download
and disk-stream, so iteration is cheap) before ever touching the real 8B target:

  (a) prefill returns logits + populated past_key_values
  (b) single-token decode with cache reuse matches a full re-prefill
  (c) multi-token (K>1) verification call with cache returns correct per-position logits
  (d) cache rollback/truncation after a simulated rejection works

Each check prints PASS/FAIL. This determines which branch kv_cache_utils.py takes.
"""

import time

import torch
from airllm import AutoModel
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache


def clone_cache(cache: DynamicCache) -> DynamicCache:
    """DynamicCache tensors aren't graph leaves, so torch.Tensor.__deepcopy__ refuses them.
    Clone by copying each layer's key/value tensors into a fresh cache instead."""
    new_cache = DynamicCache()
    for idx, layer in enumerate(cache.layers):
        new_cache.update(layer.keys.clone(), layer.values.clone(), idx)
    return new_cache

SPIKE_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPT = "The capital of France is"


def log(msg: str) -> None:
    print(f"[spike] {msg}", flush=True)


def close_logits(a: torch.Tensor, b: torch.Tensor, atol: float = 0.2) -> bool:
    # bf16 (this model's runtime dtype) has ~3 decimal digits of precision, so a parallel
    # multi-token forward and a from-scratch full-sequence forward accumulate rounding
    # differently and will not match bit-for-bit. What matters functionally is that the
    # argmax (and ideally top-k ranking) agree, not exact logit values.
    close_enough = torch.allclose(a.float(), b.float(), atol=atol, rtol=0.05)
    argmax_matches = torch.equal(a.argmax(dim=-1), b.argmax(dim=-1))
    return close_enough and argmax_matches


def main() -> None:
    log(f"loading tokenizer + AirLLM model: {SPIKE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(SPIKE_MODEL)
    model = AutoModel.from_pretrained(SPIKE_MODEL, device="cuda:0", max_seq_len=128)

    prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.to("cuda:0")
    results = {}

    # (a) prefill
    t0 = time.time()
    out = model(input_ids=prompt_ids, use_cache=True)
    prefill_time = time.time() - t0
    has_logits = hasattr(out, "logits") and out.logits is not None
    has_cache = getattr(out, "past_key_values", None) is not None
    results["a_prefill"] = has_logits and has_cache
    log(f"(a) prefill: logits={tuple(out.logits.shape) if has_logits else None} "
        f"cache_present={has_cache} time={prefill_time:.2f}s -> "
        f"{'PASS' if results['a_prefill'] else 'FAIL'}")

    if not results["a_prefill"]:
        log("cannot continue without a working prefill call. STOPPING.")
        _print_summary(results)
        return

    kv_after_prefill = out.past_key_values
    next_token_logits = out.logits[:, -1, :]
    next_token = next_token_logits.argmax(dim=-1, keepdim=True)

    # (b) single-token decode with cache reuse vs full re-prefill
    # cache_position must be passed explicitly: without it, transformers defaults to
    # range(0, seq_len) instead of range(past_len, past_len+seq_len), corrupting position
    # embeddings/attention for any call with an existing past_key_values.
    past_len = prompt_ids.shape[1]
    cache_position_b = torch.arange(past_len, past_len + next_token.shape[1], device="cuda:0")
    # clone: DynamicCache.update() mutates the cache object in place, and kv_after_prefill is
    # reused (pristine) for check (c) below, so each check needs its own copy.
    t0 = time.time()
    out_b_cached = model(input_ids=next_token, past_key_values=clone_cache(kv_after_prefill),
                          use_cache=True, cache_position=cache_position_b)
    single_time = time.time() - t0

    full_ids = torch.cat([prompt_ids, next_token], dim=1)
    out_b_full = model(input_ids=full_ids, use_cache=True)
    cached_last_logits = out_b_cached.logits[:, -1, :]
    full_last_logits = out_b_full.logits[:, -1, :]
    results["b_single_token_cache"] = close_logits(cached_last_logits, full_last_logits)
    log(f"(b) single-token cache reuse vs re-prefill match: "
        f"{'PASS' if results['b_single_token_cache'] else 'FAIL'} "
        f"(cached_time={single_time:.2f}s)")

    # (c) multi-token (K>1) verification call with cache
    # Draft tokens come from the model's own greedy continuation, not random ids: random/OOD
    # token ids push activations far outside the trained distribution and amplify ordinary
    # fp16/bf16 accumulation-order differences between a parallel cached-forward pass and a
    # from-scratch full-sequence forward pass, causing spurious mismatches unrelated to whether
    # multi-token verification itself works. Real draft tokens are always in-distribution.
    K = 4
    draft_tokens = []
    _cur = next_token
    _kv = clone_cache(kv_after_prefill)
    _pos = past_len
    for _ in range(K):
        _cp = torch.arange(_pos, _pos + 1, device="cuda:0")
        _o = model(input_ids=_cur, past_key_values=_kv, use_cache=True, cache_position=_cp)
        _kv = _o.past_key_values
        _cur = _o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        draft_tokens.append(_cur)
        _pos += 1
    draft_tokens = torch.cat(draft_tokens, dim=1)
    multi_input = torch.cat([next_token, draft_tokens], dim=1)  # K+1 new positions after prefill cache
    cache_position_c = torch.arange(past_len, past_len + multi_input.shape[1], device="cuda:0")

    t0 = time.time()
    out_c_cached = model(input_ids=multi_input, past_key_values=clone_cache(kv_after_prefill),
                          use_cache=True, cache_position=cache_position_c)
    multi_time = time.time() - t0

    full_ids_c = torch.cat([prompt_ids, multi_input], dim=1)
    out_c_full = model(input_ids=full_ids_c, use_cache=True)

    shape_ok = out_c_cached.logits.shape[1] == multi_input.shape[1]
    values_ok = close_logits(out_c_cached.logits, out_c_full.logits[:, -multi_input.shape[1]:, :])
    results["c_multi_token_verify"] = shape_ok and values_ok
    log(f"(c) multi-token verify: shape={tuple(out_c_cached.logits.shape)} "
        f"shape_ok={shape_ok} values_ok={values_ok} time={multi_time:.2f}s -> "
        f"{'PASS' if results['c_multi_token_verify'] else 'FAIL'}")

    # (d) cache rollback after simulated rejection
    results["d_cache_rollback"] = False
    if results["c_multi_token_verify"]:
        kv_after_multi = out_c_cached.past_key_values
        rollback_len = past_len + 1  # keep prefill + 1 accepted token only
        try:
            if hasattr(kv_after_multi, "crop"):
                kv_after_multi.crop(rollback_len)
                rolled_back = kv_after_multi
            else:
                rolled_back = tuple(
                    (k[:, :, :rollback_len, :], v[:, :, :rollback_len, :])
                    for k, v in kv_after_multi
                )
            probe_token = draft_tokens[:, :1]
            cache_position_d = torch.arange(rollback_len, rollback_len + 1, device="cuda:0")
            out_d = model(input_ids=probe_token, past_key_values=rolled_back, use_cache=True,
                           cache_position=cache_position_d)
            reference_ids = torch.cat([prompt_ids, next_token, probe_token], dim=1)
            out_d_full = model(input_ids=reference_ids, use_cache=True)
            results["d_cache_rollback"] = close_logits(
                out_d.logits[:, -1, :], out_d_full.logits[:, -1, :]
            )
        except Exception as exc:  # noqa: BLE001 - spike diagnostic, want to see any failure mode
            log(f"(d) rollback raised: {exc!r}")
    log(f"(d) cache rollback after rejection: "
        f"{'PASS' if results['d_cache_rollback'] else 'FAIL'}")

    _print_summary(results)


def _print_summary(results: dict) -> None:
    log("=" * 50)
    log("SUMMARY")
    for name, passed in results.items():
        log(f"  {name}: {'PASS' if passed else 'FAIL'}")
    all_core_pass = results.get("a_prefill") and results.get("c_multi_token_verify")
    if all_core_pass and results.get("d_cache_rollback"):
        log("GO: full incremental KV reuse + rollback supported. "
            "kv_cache_utils.py should implement the true-reuse path.")
    elif all_core_pass:
        log("PARTIAL GO: multi-token verify works, but rollback does not. "
            "kv_cache_utils.py should re-prefill from prompt_so_far after any rejection "
            "instead of truncating the cache in place.")
    else:
        log("NO-GO on multi-token verify: kv_cache_utils.py must fall back to full "
            "re-prefill (prompt + K draft tokens) on every verification call, no cache reuse "
            "across the draft/verify boundary.")


if __name__ == "__main__":
    main()
