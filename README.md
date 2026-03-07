# Speculative Decoding on 4GB VRAM

Token-level speculative decoding where a small draft model (Qwen2.5-3B, 4-bit)
proposes tokens that a much larger target model (Qwen2.5-7B, streamed via AirLLM)
verifies in a single forward pass, on a 4GB-VRAM GPU. Draft and target share the
same tokenizer/vocabulary (same model family), as required for rejection sampling
to compare token probabilities directly. Both checkpoints are ungated on Hugging
Face — no license click-through or gated-repo access needed.

## Key architectural constraint

AirLLM streams every transformer layer from disk on **every** forward call,
regardless of token count (~0.5-2 tok/s on a 4GB GPU). A verification call that
checks K draft tokens at once still pays one full disk sweep through the 8B
model's layers — so this project does **not** target wall-clock speedup over a
full-VRAM baseline. The primary success metric is **AirLLM forward-call count**:
baseline pays 1 call/token, speculative decoding aims to pay 1 call per accepted
batch of tokens (ideally >1 token/call). Latency/tokens-per-sec are tracked as
secondary, explicitly-caveated numbers.

## Setup

```bash
uv sync
```

Requires an NVIDIA GPU (tested on RTX 3050 Ti Laptop, 4096 MiB) with CUDA-capable
driver. `Qwen/Qwen2.5-3B` and `Qwen/Qwen2.5-7B` are both ungated, so no Hugging Face
login is required to download them.

## Build order

See `.claude` plan history / project docs for the full phase-by-phase plan.
Run `scripts/00_spike_airllm_forward.py` first — it is a go/no-go gate for the
rest of the pipeline's KV-cache-reuse assumptions.

## Implementation Risk Findings

Confirmed via `scripts/00_spike_airllm_forward.py` against `HuggingFaceTB/SmolLM2-135M-Instruct`
(a small Llama-architecture model, used for fast iteration; AirLLM handles any standard
`*ForCausalLM` generically, so this generalizes to the real Qwen2.5-7B target):

- [x] (a) prefill returns logits + populated past_key_values
- [x] (b) single-token decode with cache reuse matches full re-prefill
- [x] (c) multi-token (K>1) verification call with cache returns correct per-position logits
- [x] (d) cache rollback/truncation after simulated rejection works

**Path selected for `kv_cache_utils.py`: full incremental KV reuse ("GO").**

Two implementation details were required to get a clean GO, both encoded permanently in
`src/specdec/kv_cache_utils.py`:

1. **`cache_position` must always be passed explicitly** alongside `past_key_values`.
   Without it, transformers defaults to `range(0, seq_len)` instead of
   `range(past_len, past_len + seq_len)`, silently corrupting position embeddings/attention
   for any call carrying an existing cache — this looked like a fundamental multi-token
   verification failure until traced to this.
2. **`DynamicCache.update()` mutates its cache object in place.** Any cache that needs to be
   reused for more than one downstream call (e.g. the same post-prefill cache used for both a
   single-token check and a multi-token verification check) must be cloned first — `clone_cache()`
   does this manually since `torch.Tensor.__deepcopy__` refuses non-leaf tensors.

One more thing worth recording: logit-value comparisons must tolerate ordinary bf16 rounding
noise (this model runs in bf16) — a parallel multi-token forward and a from-scratch full-sequence
forward can differ by ~0.1 in raw logit magnitude (~1% relative) while agreeing exactly on
argmax and full top-k ranking. The spike's correctness check requires both a loose numeric
tolerance *and* an exact argmax match, rather than a tight `allclose` alone.
