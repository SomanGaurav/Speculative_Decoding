"""Small QA/reasoning benchmark subsets for the eval harness. Kept intentionally
tiny (config-controlled `num_examples`) since each AirLLM call takes ~10-13s on
this hardware -- see config/eval.yaml.

Each loader yields (prompt: str, reference_answer: str) pairs.
"""

import itertools

from datasets import load_dataset

MMLU_CHOICE_LETTERS = ["A", "B", "C", "D"]


def load_mmlu(subset: str, num_examples: int):
    dataset = load_dataset("cais/mmlu", subset, split="test")
    for i in range(min(num_examples, len(dataset))):
        row = dataset[i]
        choices_text = "\n".join(
            f"{letter}. {choice}" for letter, choice in zip(MMLU_CHOICE_LETTERS, row["choices"])
        )
        prompt = f"{row['question']}\n{choices_text}\nAnswer:"
        reference = MMLU_CHOICE_LETTERS[row["answer"]]
        yield prompt, reference


def load_gsm8k(num_examples: int):
    dataset = load_dataset("openai/gsm8k", "main", split="test")
    for i in range(min(num_examples, len(dataset))):
        row = dataset[i]
        prompt = f"Question: {row['question']}\nAnswer:"
        reference = row["answer"].split("####")[-1].strip()
        yield prompt, reference


def load_triviaqa(num_examples: int):
    dataset = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation",
                            streaming=True)
    for row in itertools.islice(dataset, num_examples):
        prompt = f"Question: {row['question']}\nAnswer:"
        reference = row["answer"]["value"]
        yield prompt, reference


LOADERS = {
    "mmlu": lambda cfg: load_mmlu(cfg["subset"], cfg["num_examples"]),
    "gsm8k": lambda cfg: load_gsm8k(cfg["num_examples"]),
    "triviaqa": lambda cfg: load_triviaqa(cfg["num_examples"]),
}


def load_eval_examples(datasets_config: dict):
    """Yields (dataset_name, prompt, reference_answer) across all configured datasets."""
    for name, cfg in datasets_config.items():
        for prompt, reference in LOADERS[name](cfg):
            yield name, prompt, reference
