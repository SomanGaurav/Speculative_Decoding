"""Run the speculative decoding pipeline over the configured eval subset, for each
speculative_variants entry in config/eval.yaml (adaptive vs. fixed-k ablation), and
dump per-example metrics to results/raw/speculative_<variant>_<timestamp>.jsonl.
"""

import json
import sys
import time
from pathlib import Path

import yaml
from transformers import AutoTokenizer

sys.path.insert(0, "src")
from specdec.adaptive_length import LengthConfig
from specdec.datasets_loader import load_eval_examples
from specdec.draft_model import load_draft_model
from specdec.speculative_loop import generate_speculative
from specdec.target_model import load_target_model


def main() -> None:
    with open("config/models.yaml") as f:
        models_cfg = yaml.safe_load(f)
    with open("config/eval.yaml") as f:
        eval_cfg = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(models_cfg["draft"]["model_id"])
    print("[run-speculative] loading draft model ...")
    draft_model, _ = load_draft_model(
        models_cfg["draft"]["model_id"], quant_bits=models_cfg["draft"]["quant_bits"],
        device=models_cfg["draft"]["device"],
    )
    print("[run-speculative] loading target model (AirLLM) ...")
    target_model = load_target_model(
        models_cfg["target"]["model_id"], compression=models_cfg["target"]["airllm_compression"],
        max_seq_len=512,
    )

    examples = list(load_eval_examples(eval_cfg["datasets"]))
    print(f"[run-speculative] {len(examples)} examples loaded")

    timestamp = int(time.time())
    Path("results/raw").mkdir(parents=True, exist_ok=True)

    for variant in eval_cfg["speculative_variants"]:
        length_config = LengthConfig(
            mode=variant["mode"],
            max_k=variant["max_k"],
            entropy_threshold=variant["entropy_threshold"] or 0.0,
        )
        out_path = Path(f"results/raw/speculative_{variant['name']}_{timestamp}.jsonl")
        print(f"[run-speculative] variant={variant['name']} -> {out_path}")

        with open(out_path, "w") as f:
            for dataset_name, prompt, reference in examples:
                prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
                tokens, metrics = generate_speculative(
                    prompt_ids, draft_model, target_model,
                    max_new_tokens=eval_cfg["max_new_tokens"], length_config=length_config,
                    eos_token_id=tokenizer.eos_token_id,
                )
                generated_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
                record = {
                    "dataset": dataset_name,
                    "variant": variant["name"],
                    "prompt": prompt,
                    "reference": reference,
                    "generated": generated_text,
                    **metrics.to_dict(),
                }
                f.write(json.dumps(record) + "\n")
                f.flush()
                print(f"  [{dataset_name}] calls={metrics.airllm_calls} "
                      f"tokens/call={metrics.tokens_per_call:.2f} "
                      f"generated={generated_text!r}")


if __name__ == "__main__":
    main()
