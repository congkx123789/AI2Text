import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def iter_records(path: Path) -> Iterable[Dict]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        data = json.load(path.open())
        if isinstance(data, list):
            for item in data:
                yield item
        elif isinstance(data, dict) and "results" in data:
            for item in data["results"]:
                yield item
        else:
            raise ValueError(f"Unexpected JSON structure in {path}")


def extract_sdi(rec: Dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    sub_keys = ["substitution", "substitutions", "subs", "S"]
    del_keys = ["deletion", "deletions", "dels", "D"]
    ins_keys = ["insertion", "insertions", "ins", "I"]
    sub = next((rec.get(k) for k in sub_keys if k in rec), None)
    dele = next((rec.get(k) for k in del_keys if k in rec), None)
    ins = next((rec.get(k) for k in ins_keys if k in rec), None)
    return sub, dele, ins


def extract_duration(rec: Dict) -> Optional[float]:
    for key in ["duration", "audio_duration", "audio_len", "length", "len"]:
        if key in rec:
            return rec.get(key)
    return None


def load_model(path: Path) -> Dict[int, Dict]:
    model = {}
    for rec in iter_records(path):
        sid = rec.get("sample_id")
        if sid is None:
            continue
        model[sid] = rec
    return model


def build_frame(model_a: Dict[int, Dict], model_b: Dict[int, Dict]) -> pd.DataFrame:
    common_ids = sorted(set(model_a) & set(model_b))
    rows: List[Dict] = []
    for sid in common_ids:
        ra, rb = model_a[sid], model_b[sid]
        rows.append(
            {
                "sample_id": sid,
                "wer_a": ra.get("wer"),
                "wer_b": rb.get("wer"),
                "cer_a": ra.get("cer"),
                "cer_b": rb.get("cer"),
                "duration": extract_duration(ra) or extract_duration(rb),
                "sub_a": extract_sdi(ra)[0],
                "del_a": extract_sdi(ra)[1],
                "ins_a": extract_sdi(ra)[2],
                "sub_b": extract_sdi(rb)[0],
                "del_b": extract_sdi(rb)[1],
                "ins_b": extract_sdi(rb)[2],
            }
        )
    return pd.DataFrame(rows)


def plot_scatter(df: pd.DataFrame, out_prefix: Path) -> None:
    ax = sns.scatterplot(data=df, x="wer_a", y="wer_b", alpha=0.4)
    max_val = float(df[["wer_a", "wer_b"]].max().max())
    ax.plot([0, max_val], [0, max_val], "r--", label="y = x")
    ax.set_title("Sample-wise WER comparison (below line = Model B better)")
    ax.set_xlabel("WER Model A")
    ax.set_ylabel("WER Model B")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".scatter.png"))
    plt.close()


def plot_box(df: pd.DataFrame, out_prefix: Path) -> None:
    melted = df.melt(
        value_vars=["wer_a", "wer_b"], var_name="model", value_name="wer"
    )
    ax = sns.boxplot(data=melted, x="model", y="wer", palette="Set2")
    ax.set_title("WER distribution")
    ax.set_xlabel("")
    ax.set_ylabel("WER")
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".box.png"))
    plt.close()


def plot_stacked_bar(df: pd.DataFrame, out_prefix: Path) -> None:
    # Sum S/D/I if present; skip if all missing.
    sums = {}
    for label, cols in [("Model A", ["sub_a", "del_a", "ins_a"]), ("Model B", ["sub_b", "del_b", "ins_b"])]:
        vals = [df[c].dropna().sum() for c in cols]
        if all(math.isclose(v, 0.0) for v in vals):
            vals = None
        sums[label] = vals
    if any(v is None for v in sums.values()):
        return
    error_df = pd.DataFrame(
        {
            "Model": ["Model A", "Model B"],
            "Substitution": sums["Model A"][0],
            "Deletion": sums["Model A"][1],
            "Insertion": sums["Model A"][2],
        }
    )
    error_df.loc[1, ["Substitution", "Deletion", "Insertion"]] = sums["Model B"]
    error_df.set_index("Model").plot(
        kind="bar",
        stacked=True,
        color=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )
    plt.title("Error type breakdown (S/D/I)")
    plt.ylabel("Total count")
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".sdi.png"))
    plt.close()


def plot_duration(df: pd.DataFrame, out_prefix: Path) -> None:
    if df["duration"].isna().all():
        return
    bins = [0, 3, 6, 9, 12, math.inf]
    labels = ["0-3s", "3-6s", "6-9s", "9-12s", ">12s"]
    df = df.copy()
    df["duration_bin"] = pd.cut(df["duration"], bins=bins, labels=labels)
    grouped = (
        df.groupby("duration_bin")[["wer_a", "wer_b"]]
        .mean()
        .reset_index()
        .melt(id_vars="duration_bin", var_name="model", value_name="avg_wer")
    )
    ax = sns.lineplot(
        data=grouped, x="duration_bin", y="avg_wer", hue="model", marker="o"
    )
    ax.set_title("Average WER by duration bucket")
    ax.set_xlabel("Duration bucket")
    ax.set_ylabel("Average WER")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".duration.png"))
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot visual comparisons for two ASR result files."
    )
    parser.add_argument("--model-a", required=True, help="Path to model A results (json/jsonl).")
    parser.add_argument("--model-b", required=True, help="Path to model B results (json/jsonl).")
    parser.add_argument(
        "--out-prefix",
        default="comparison",
        help="Output file prefix for generated plots.",
    )
    args = parser.parse_args()

    path_a = Path(args.model_a)
    path_b = Path(args.model_b)
    out_prefix = Path(args.out_prefix)

    model_a = load_model(path_a)
    model_b = load_model(path_b)
    if not model_a or not model_b:
        raise SystemExit("No records found in one of the files.")

    df = build_frame(model_a, model_b)
    if df.empty:
        raise SystemExit("No common sample_ids to compare.")

    sns.set(style="whitegrid")
    plot_scatter(df, out_prefix)
    plot_box(df, out_prefix)
    plot_stacked_bar(df, out_prefix)
    plot_duration(df, out_prefix)

    summary = {
        "total_common": len(df),
        "avg_wer_a": float(df["wer_a"].mean()),
        "avg_wer_b": float(df["wer_b"].mean()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

