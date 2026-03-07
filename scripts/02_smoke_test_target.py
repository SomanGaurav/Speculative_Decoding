"""Smoke test: load the AirLLM-streamed target model, run one prefill + one verify()
call on a short prompt with a handful of draft tokens, confirm logits shape, confirm
the call counter increments correctly, and time both calls to get a real per-call
latency baseline on this hardware.

Qwen/Qwen2.5-7B is ungated, so no Hugging Face login is required.
"""

import sys
import time

import torch
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, "src")
from specdec.metrics import CallCounter
from specdec.target_model import load_target_model, prefill, verify


def main() -> None:
    with open("config/models.yaml") as f:
        cfg = yaml.safe_load(f)["target"]

    print(f"[smoke-target] loading {cfg['model_id']} via AirLLM "
          f"(compression={cfg['airllm_compression']}) ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"])
    model = load_target_model(cfg["model_id"], compression=cfg["airllm_compression"],
                               max_seq_len=128)

    prompt_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids.to("cuda:0")
    counter = CallCounter()

    t0 = time.time()
    prefill_result = prefill(model, prompt_ids, counter)
    prefill_time = time.time() - t0
    print(f"[smoke-target] prefill: probs shape={tuple(prefill_result.probs.shape)} "
          f"time={prefill_time:.2f}s calls={counter.count}")
    assert counter.count == 1, "prefill must count as exactly 1 AirLLM call"

    next_token = prefill_result.probs.argmax(dim=-1, keepdim=True)
    draft_tokens = torch.randint(low=0, high=tokenizer.vocab_size, size=(1, 3), device="cuda:0")
    verify_input = torch.cat([next_token, draft_tokens], dim=1)

    t0 = time.time()
    verify_result = verify(model, verify_input, prompt_ids.shape[1], prefill_result.cache, counter)
    verify_time = time.time() - t0
    print(f"[smoke-target] verify (K={verify_input.shape[1]}): "
          f"probs shape={tuple(verify_result.probs.shape)} time={verify_time:.2f}s "
          f"calls={counter.count}")
    assert counter.count == 2, "verify() must increment the call counter by exactly 1"
    assert verify_result.probs.shape[0] == verify_input.shape[1], \
        "verify() must return one probability row per verified position"

    print(f"[smoke-target] per-call latency baseline on this hardware: "
          f"prefill={prefill_time:.2f}s, verify(K={verify_input.shape[1]})={verify_time:.2f}s")


if __name__ == "__main__":
    main()
