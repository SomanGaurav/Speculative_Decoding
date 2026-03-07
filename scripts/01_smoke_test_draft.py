"""Smoke test: load the draft model in 4-bit, run draft_step, sanity-check output
and peak VRAM usage against the ~2-2.5GB budget it needs to leave headroom for
AirLLM's single active layer on a 4GB GPU.

Qwen/Qwen2.5-3B is ungated, so no Hugging Face login is required.
"""

import sys

import torch
import yaml

sys.path.insert(0, "src")
from specdec.adaptive_length import LengthConfig
from specdec.draft_model import draft_step, load_draft_model


def main() -> None:
    with open("config/models.yaml") as f:
        cfg = yaml.safe_load(f)["draft"]

    print(f"[smoke-draft] loading {cfg['model_id']} in {cfg['quant_bits']}-bit ...")
    torch.cuda.reset_peak_memory_stats()
    model, tokenizer = load_draft_model(cfg["model_id"], quant_bits=cfg["quant_bits"], device=cfg["device"])

    prompt = "The capital of France is"
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(cfg["device"])

    with torch.no_grad():
        prefill_out = model(input_ids=prompt_ids, use_cache=True)
    cache = prefill_out.past_key_values
    first_token = prefill_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    length_config = LengthConfig(mode="entropy_threshold", max_k=6, entropy_threshold=2.0)
    result = draft_step(model, first_token, cache, length_config)

    full_ids = torch.cat([prompt_ids, first_token, result.tokens.unsqueeze(0)], dim=1)
    text = tokenizer.decode(full_ids[0], skip_special_tokens=True)

    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"[smoke-draft] generated: {text!r}")
    print(f"[smoke-draft] proposed {result.tokens.shape[0]} tokens, entropies={result.entropies}")
    print(f"[smoke-draft] peak VRAM: {peak_gb:.2f} GB")
    if peak_gb > 2.5:
        print("[smoke-draft] WARNING: draft model alone exceeds the ~2.5GB budget "
              "reserved to leave headroom for AirLLM's single active layer.")


if __name__ == "__main__":
    main()
