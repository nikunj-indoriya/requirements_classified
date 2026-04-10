# =============================================================================
# analysis.py — Post-experiment analysis and visualisation
# =============================================================================
# Usage:
#   python analysis.py --dataset promise
#   python analysis.py --dataset crowdre  --labeling_mode wikidominer
#   python analysis.py --dataset secreq   --labeling_mode hybrid
# =============================================================================

import os
import ast
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_merge(dataset_name, labeling_mode="centroid", max_per_class=None):
    tag        = f"{dataset_name}_sub{max_per_class}" if max_per_class else dataset_name
    unsup_path = f"results/{tag}_unsupervised_{labeling_mode}_full.csv"
    sup_path   = f"results/{tag}_supervised_full.csv"

    if not os.path.exists(unsup_path):
        raise FileNotFoundError(
            f"Missing results file: {unsup_path}\n"
            f"Run: python run_unsupervised.py --dataset {dataset_name} "
            f"--path data/<file> --labeling_mode {labeling_mode}"
        )
    if not os.path.exists(sup_path):
        raise FileNotFoundError(
            f"Missing results file: {sup_path}\n"
            f"Run: python run_supervised.py --dataset {dataset_name} --path data/<file>"
        )

    unsup = pd.read_csv(unsup_path)
    sup   = pd.read_csv(sup_path)

    sup["embedding"]  = "logistic_regression"
    sup["clustering"] = "supervised"
    if "alpha" not in unsup.columns:
        unsup["alpha"] = None
    sup["alpha"] = None

    return pd.concat([unsup, sup], ignore_index=True)


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_summary_stats(df, output_dir, labeling_mode):
    group_cols = ["embedding", "clustering", "k", "alpha"] if labeling_mode == "hybrid" \
                 else ["embedding", "clustering", "k"]
    summary = df.groupby(group_cols)["f1"].agg(["mean", "std"]).reset_index()
    summary.to_csv(os.path.join(output_dir, "summary_stats.csv"), index=False)
    return summary


def plot_degradation(summary, output_dir, labeling_mode):
    # Per-embedding / per-alpha degradation (KMeans)
    plt.figure(figsize=(8, 6))
    if labeling_mode == "hybrid":
        for alpha in sorted(summary["alpha"].dropna().unique()):
            s = summary[(summary["clustering"] == "kmeans") & (summary["alpha"] == alpha)]
            if not s.empty:
                plt.plot(s["k"], s["mean"], label=f"alpha={alpha}")
    else:
        for emb in summary["embedding"].unique():
            s = summary[(summary["embedding"] == emb) & (summary["clustering"] == "kmeans")]
            if not s.empty:
                plt.plot(s["k"], s["mean"], label=emb)
    plt.xlabel("Number of Classes (k)")
    plt.ylabel("Mean Macro F1")
    plt.title("Degradation Curve — KMeans")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "degradation_kmeans.png"))
    plt.close()

    # Unsupervised vs supervised comparison
    plt.figure(figsize=(8, 6))
    for emb in summary["embedding"].unique():
        s = summary[(summary["embedding"] == emb) & (summary["clustering"] == "kmeans")]
        if not s.empty:
            plt.plot(s["k"], s["mean"], label=emb)
    sup_s = summary[summary["embedding"] == "logistic_regression"]
    if not sup_s.empty:
        plt.plot(sup_s["k"], sup_s["mean"], linestyle="--", linewidth=2, label="Logistic Regression")
    plt.xlabel("Number of Classes (k)")
    plt.ylabel("Mean Macro F1")
    plt.title("Unsupervised vs Supervised")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "degradation_comparison.png"))
    plt.close()


def alpha_analysis(summary, output_dir):
    alpha_df = summary.dropna(subset=["alpha"])
    if alpha_df.empty:
        return
    records = [{"alpha": a, "avg_f1": alpha_df[alpha_df["alpha"] == a]["mean"].mean()}
               for a in sorted(alpha_df["alpha"].unique())]
    result = pd.DataFrame(records)
    result.to_csv(os.path.join(output_dir, "alpha_analysis.csv"), index=False)
    plt.figure()
    plt.plot(result["alpha"], result["avg_f1"], marker="o")
    plt.xlabel("Alpha"); plt.ylabel("Average F1"); plt.title("Alpha vs Performance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "alpha_plot.png"))
    plt.close()


def compute_slopes(summary, output_dir):
    records = []
    for emb in summary["embedding"].unique():
        s = summary[summary["embedding"] == emb]
        X, y = s["k"].values.reshape(-1, 1), s["mean"].values
        m = LinearRegression().fit(X, y)
        records.append({"method": emb, "slope": m.coef_[0], "r2": m.score(X, y)})
    pd.DataFrame(records).to_csv(os.path.join(output_dir, "degradation_slopes.csv"), index=False)


def compute_gap(summary, output_dir):
    records = []
    for k in sorted(summary["k"].unique()):
        sup_val = summary[(summary["embedding"] == "logistic_regression") & (summary["k"] == k)]["mean"].values
        if not len(sup_val):
            continue
        for _, row in summary[(summary["embedding"] != "logistic_regression") & (summary["k"] == k)].iterrows():
            records.append({"k": k, "unsupervised_method": row["embedding"], "gap": sup_val[0] - row["mean"]})
    pd.DataFrame(records).to_csv(os.path.join(output_dir, "performance_gap.csv"), index=False)


def compute_robustness(summary, output_dir):
    records = []
    k_max = summary["k"].max()
    for emb in summary["embedding"].unique():
        f2   = summary[(summary["embedding"] == emb) & (summary["k"] == 2)]["mean"].values
        fmax = summary[(summary["embedding"] == emb) & (summary["k"] == k_max)]["mean"].values
        if len(f2) and len(fmax):
            records.append({"method": emb, "robustness_ratio": fmax[0] / f2[0]})
    pd.DataFrame(records).to_csv(os.path.join(output_dir, "robustness_ratio.csv"), index=False)


def compare_clustering(summary, output_dir):
    records = []
    for emb in summary["embedding"].unique():
        km  = summary[(summary["embedding"] == emb) & (summary["clustering"] == "kmeans")]
        hac = summary[(summary["embedding"] == emb) & (summary["clustering"] == "hac")]
        if not km.empty and not hac.empty:
            merged = pd.merge(km, hac, on="k", suffixes=("_kmeans", "_hac"))
            for _, row in merged.iterrows():
                records.append({"embedding": emb, "k": row["k"],
                                "difference": row["mean_kmeans"] - row["mean_hac"]})
    pd.DataFrame(records).to_csv(os.path.join(output_dir, "clustering_comparison.csv"), index=False)


def compute_binary_class_difficulty(df, output_dir):
    binary_df = df[df["k"] == 2]
    records   = []
    for method in binary_df["embedding"].unique():
        class_scores = {}
        for _, row in binary_df[binary_df["embedding"] == method].iterrows():
            for c in ast.literal_eval(row["class_subset"]):
                class_scores.setdefault(c, []).append(row["f1"])
        for c, scores in class_scores.items():
            records.append({"method": method, "class": c, "avg_binary_f1": np.mean(scores)})
    pd.DataFrame(records).to_csv(os.path.join(output_dir, "binary_class_difficulty.csv"), index=False)


def compute_ranking(summary, output_dir):
    records = []
    for k in sorted(summary["k"].unique()):
        for rank, (_, row) in enumerate(summary[summary["k"] == k].sort_values("mean", ascending=False).iterrows(), 1):
            records.append({"k": k, "method": row["embedding"], "rank": rank, "mean_f1": row["mean"]})
    pd.DataFrame(records).to_csv(os.path.join(output_dir, "ranking_per_k.csv"), index=False)


# =============================================================================
# MAIN
# =============================================================================

def main(dataset_name, labeling_mode="centroid", max_per_class=None):
    tag       = f"{dataset_name}_sub{max_per_class}" if max_per_class else dataset_name
    dir_map   = {"centroid": "analysis", "wikidominer": "analysis_wikidominer", "hybrid": "analysis_hybrid"}
    output_dir = f"results/{dir_map[labeling_mode]}/{tag}"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading and merging data...")
    df = load_and_merge(dataset_name, labeling_mode, max_per_class)

    print("Computing summary statistics...")
    summary = compute_summary_stats(df, output_dir, labeling_mode)

    print("Plotting degradation curves...")
    plot_degradation(summary, output_dir, labeling_mode)

    print("Computing degradation slopes...")
    compute_slopes(summary, output_dir)

    print("Computing performance gap...")
    compute_gap(summary, output_dir)

    print("Computing robustness ratio...")
    compute_robustness(summary, output_dir)

    print("Comparing clustering methods...")
    compare_clustering(summary, output_dir)

    print("Computing binary class difficulty...")
    compute_binary_class_difficulty(df, output_dir)

    print("Computing ranking per k...")
    compute_ranking(summary, output_dir)

    if labeling_mode == "hybrid":
        print("Running alpha analysis...")
        alpha_analysis(summary, output_dir)

    print(f"\nAnalysis complete → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",       type=str, required=True)
    parser.add_argument("--labeling_mode", type=str, default="centroid",
                        choices=["centroid", "wikidominer", "hybrid"])
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Must match the value used when running experiments")
    args = parser.parse_args()
    main(args.dataset, args.labeling_mode, args.max_per_class)
