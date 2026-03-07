"""Draft model: Llama-3.2-3B, 4-bit quantized (fp16 alone needs ~7GB, already over
the 4GB VRAM budget on its own), autoregressively proposing tokens with per-token
entropy tracked for the adaptive speculation-length heuristic.
"""

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .adaptive_length import LengthConfig, should_continue_drafting
from .kv_cache_utils import cache_position_for, clone_cache


@dataclass
class DraftResult:
    tokens: torch.Tensor       # (K,) proposed token ids
    probs: torch.Tensor        # (K, vocab) draft model's per-position probabilities
    entropies: list[float]     # per-token softmax entropy, for metrics/analysis


def load_draft_model(model_id: str, quant_bits: int = 4, device: str = "cuda:0"):
    quant_config = BitsAndBytesConfig(
        load_in_4bit=quant_bits == 4,
        load_in_8bit=quant_bits == 8,
        bnb_4bit_compute_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant_config, device_map=device,
    )
    model.eval()
    return model, tokenizer


def _entropy(probs: torch.Tensor) -> float:
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum().item()


@torch.no_grad()
def draft_step(
    model,
    start_token: torch.Tensor,
    cache,
    length_config: LengthConfig,
) -> DraftResult:
    """Autoregressively propose up to length_config.max_k tokens, starting from
    `start_token` (shape (1, 1) — the most recently committed token) against `cache`
    (already covering every position before start_token). Stops early per the
    adaptive-length heuristic. Runs fully in VRAM against the (fast) draft model,
    not the disk-streamed target; `cache` is cloned first, so the caller's copy is
    left untouched."""
    device = start_token.device
    working_cache = clone_cache(cache)

    tokens, probs_list, entropies = [], [], []
    cur_input = start_token
    step = 0
    while True:
        cache_position = cache_position_for(working_cache.get_seq_length(), cur_input.shape[1], device)
        out = model(input_ids=cur_input, past_key_values=working_cache, use_cache=True,
                     cache_position=cache_position)
        working_cache = out.past_key_values
        logits = out.logits[:, -1, :]
        token_probs = torch.softmax(logits, dim=-1)
        next_token = token_probs.argmax(dim=-1, keepdim=True)
        entropy = _entropy(token_probs.squeeze(0))

        tokens.append(next_token)
        probs_list.append(token_probs)
        entropies.append(entropy)

        step += 1
        if not should_continue_drafting(step - 1, entropy, length_config):
            break
        cur_input = next_token

    return DraftResult(
        tokens=torch.cat(tokens, dim=1).squeeze(0),
        probs=torch.cat(probs_list, dim=0),
        entropies=entropies,
    )
