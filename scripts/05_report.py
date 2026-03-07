"""Aggregate results/raw/*.jsonl into comparison tables/plots in results/figures/.

Primary result: mean AirLLM forward calls per example, baseline vs. each speculative
variant, with % reduction -- this is the project's headline metric (see README).
Secondary: latency/tokens-per-second (explicitly caveated, not a wall-clock speedup
claim), accept-rate vs. entropy-threshold sensitivity, correctness/accuracy sanity
check, and resource usage (confirming the 4GB VRAM budget was respected).
"""

import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_results(raw_dir: str = "results/raw") -> pd.DataFrame:
    rows = []
    for path in glob.glob(f"{raw_dir}/*.jsonl"):
        with open(path) as f:
            for line in f:
                rows.append(json.loads(line))
    if not rows:
        raise FileNotFoundError(f"no .jsonl files found in {raw_dir}")
    return pd.DataFrame(rows)


def is_correct(row) -> bool:
    reference = str(row["reference"]).strip().lower()
    generated = str(row["generated"]).strip().lower()
    return reference in generated


def main() -> None:
    df = load_results()
    df["correct"] = df.apply(is_correct, axis=1)

    out_dir = Path("results/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- primary table: AirLLM call count ---
    call_table = df.groupby("variant").agg(
        mean_airllm_calls=("airllm_calls", "mean"),
        mean_generated_tokens=("generated_tokens", "mean"),
        mean_tokens_per_call=("tokens_per_call", "mean"),
        n_examples=("airllm_calls", "count"),
    ).reset_index()

    if "baseline" in call_table["variant"].values:
        baseline_calls = call_table.loc[call_table["variant"] == "baseline", "mean_airllm_calls"].iloc[0]
        call_table["pct_call_reduction_vs_baseline"] = (
            (baseline_calls - call_table["mean_airllm_calls"]) / baseline_calls * 100
        )
    call_table_path = out_dir / "call_count_table.csv"
    call_table.to_csv(call_table_path, index=False)
    print("=== PRIMARY: AirLLM forward-call count (baseline vs. speculative variants) ===")
    print(call_table.to_string(index=False))

    # --- secondary table: latency / tokens-per-second (caveated) ---
    latency_table = df.groupby("variant").agg(
        mean_wall_clock_seconds=("wall_clock_seconds", "mean"),
        mean_tokens_per_second=("tokens_per_second", "mean"),
    ).reset_index()
    latency_table.to_csv(out_dir / "latency_table.csv", index=False)
    print("\n=== SECONDARY (informative only, not a wall-clock speedup claim): latency ===")
    print(latency_table.to_string(index=False))

    # --- accept-rate / entropy sensitivity ---
    spec_df = df[df["variant"] != "baseline"]
    if not spec_df.empty:
        accept_table = spec_df.groupby("variant").agg(
            mean_accepted_tokens=("accepted_tokens", "mean"),
            mean_entropy=("mean_entropy", "mean"),
        ).reset_index()
        accept_table.to_csv(out_dir / "accept_rate_table.csv", index=False)
        print("\n=== Accept-rate / entropy sensitivity across speculative variants ===")
        print(accept_table.to_string(index=False))

        fig, ax = plt.subplots()
        ax.bar(accept_table["variant"], accept_table["mean_accepted_tokens"])
        ax.set_ylabel("mean accepted tokens per example")
        ax.set_title("Accepted tokens by speculative variant")
        fig.tight_layout()
        fig.savefig(out_dir / "accept_rate.png")
        plt.close(fig)

    # --- correctness sanity check ---
    correctness_table = df.groupby("variant")["correct"].mean().reset_index()
    correctness_table.columns = ["variant", "accuracy"]
    correctness_table.to_csv(out_dir / "correctness_table.csv", index=False)
    print("\n=== Correctness sanity check (speculative should track baseline's accuracy) ===")
    print(correctness_table.to_string(index=False))

    # --- resource usage ---
    resource_table = df.groupby("variant").agg(
        peak_vram_gb=("peak_vram_bytes", lambda s: s.max() / 1e9),
        peak_ram_gb=("peak_ram_bytes", lambda s: s.max() / 1e9),
    ).reset_index()
    resource_table.to_csv(out_dir / "resource_table.csv", index=False)
    print("\n=== Resource usage (confirming the 4GB VRAM budget) ===")
    print(resource_table.to_string(index=False))

    # --- headline plot: call count reduction ---
    fig, ax = plt.subplots()
    ax.bar(call_table["variant"], call_table["mean_airllm_calls"])
    ax.set_ylabel("mean AirLLM forward calls per example")
    ax.set_title("AirLLM call count: baseline vs. speculative variants")
    fig.tight_layout()
    fig.savefig(out_dir / "call_count.png")
    plt.close(fig)

    print(f"\n[report] tables + plots written to {out_dir}/")


if __name__ == "__main__":
    main()
