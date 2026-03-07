"""Run the AirLLM-only baseline over the same eval subset used by
03_run_speculative.py, dumping per-example metrics to
results/raw/baseline_<timestamp>.jsonl. This is the reference point the
speculative pipeline's call-count reduction is measured against.
"""

import json
import sys
import time
from pathlib import Path

import yaml
from transformers import AutoTokenizer

sys.path.insert(0, "src")
from specdec.baseline_loop import generate_baseline
from specdec.datasets_loader import load_eval_examples
from specdec.target_model import load_target_model


def main() -> None:
    with open("config/models.yaml") as f:
        models_cfg = yaml.safe_load(f)
    with open("config/eval.yaml") as f:
        eval_cfg = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(models_cfg["target"]["model_id"])
    print("[run-baseline] loading target model (AirLLM) ...")
    target_model = load_target_model(
        models_cfg["target"]["model_id"], compression=models_cfg["target"]["airllm_compression"],
        max_seq_len=512,
    )

    examples = list(load_eval_examples(eval_cfg["datasets"]))
    print(f"[run-baseline] {len(examples)} examples loaded")

    timestamp = int(time.time())
    Path("results/raw").mkdir(parents=True, exist_ok=True)
    out_path = Path(f"results/raw/baseline_{timestamp}.jsonl")
    print(f"[run-baseline] -> {out_path}")

    with open(out_path, "w") as f:
        for dataset_name, prompt, reference in examples:
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda:0")
            tokens, metrics = generate_baseline(
                prompt_ids, target_model, max_new_tokens=eval_cfg["max_new_tokens"],
                eos_token_id=tokenizer.eos_token_id,
            )
            generated_text = tokenizer.decode(tokens[0], skip_special_tokens=True)
            record = {
                "dataset": dataset_name,
                "variant": "baseline",
                "prompt": prompt,
                "reference": reference,
                "generated": generated_text,
                **metrics.to_dict(),
            }
            f.write(json.dumps(record) + "\n")
            f.flush()
            print(f"  [{dataset_name}] calls={metrics.airllm_calls} "
                  f"generated={generated_text!r}")


if __name__ == "__main__":
    main()
