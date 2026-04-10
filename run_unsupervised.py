# =============================================================================
# run_unsupervised.py — Unsupervised requirements classification experiments
# =============================================================================
# Usage:
#   python run_unsupervised.py --dataset promise  --path data/PROMISE_exp.arff
#   python run_unsupervised.py --dataset crowdre  --path data/requirements.csv
#   python run_unsupervised.py --dataset secreq   --path data/secreq.csv
#   python run_unsupervised.py --dataset promise  --path data/PROMISE_exp.arff --labeling_mode wikidominer
#   python run_unsupervised.py --dataset promise  --path data/PROMISE_exp.arff --labeling_mode hybrid
# =============================================================================

import argparse
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import normalize

from core import (
    load_dataset, build_embedder, get_embedding, generate_all_combinations,
    subsample_dataset,
    run_kmeans, run_hac, run_spectral, run_dbscan, run_hdbscan,
    compute_cluster_centroids, compute_class_centroids,
    elimination_label_assignment, map_clusters_to_labels,
    build_wikidominer_label_prototypes, assign_clusters_wikidominer,
    compute_macro_metrics, ExperimentLogger, ALL_EMBEDDING_MODELS,
)

CLUSTERING_METHODS = ["kmeans", "hac", "spectral", "dbscan", "hdbscan"]
DBSCAN_GRID    = {"eps": [0.3, 0.5, 0.7, 1.0], "min_samples": [3, 5, 10]}
HDBSCAN_GRID   = {"min_cluster_size": [3, 5, 10, 15]}


def output_file(dataset_name, labeling_mode):
    return f"results/{dataset_name}_unsupervised_{labeling_mode}_full.csv"


def cluster(method, embeddings, k):
    """Run a fixed-k clustering and return labels (or None if it fails)."""
    if method == "kmeans":   return run_kmeans(embeddings, k)
    if method == "hac":      return run_hac(embeddings, k)
    if method == "spectral": return run_spectral(embeddings, k)
    raise ValueError(f"Use tune_density for {method}")


def tune_density(method, embeddings, k, get_assign_fn):
    """Grid-search over density-based params; return best cluster_labels or None."""
    best_labels, best_f1 = None, -1
    grid = (
        [(eps, ms) for eps in DBSCAN_GRID["eps"] for ms in DBSCAN_GRID["min_samples"]]
        if method == "dbscan"
        else [(mcs,) for mcs in HDBSCAN_GRID["min_cluster_size"]]
    )

    for params in grid:
        lbl = run_dbscan(embeddings, *params) if method == "dbscan" else run_hdbscan(embeddings, *params)
        unique = set(lbl) - {-1}
        if len(unique) != k:
            continue
        centroids = compute_cluster_centroids(embeddings, lbl, k)
        assignment = get_assign_fn(centroids)
        predicted  = map_clusters_to_labels(lbl, assignment)
        f1 = compute_macro_metrics(np.arange(len(lbl)), predicted)["f1"]  # dummy; just for ranking
        if f1 > best_f1:
            best_f1, best_labels = f1, lbl

    return best_labels


def get_assignment(labeling_mode, alpha, subset_emb, subset_labels,
                   cluster_centroids, label_prototypes, combo, k):
    if labeling_mode == "centroid":
        class_centroids = compute_class_centroids(subset_emb, subset_labels, list(range(k)))
        return elimination_label_assignment(cluster_centroids, class_centroids)

    if labeling_mode == "wikidominer":
        return assign_clusters_wikidominer(cluster_centroids, label_prototypes[combo])

    # hybrid
    class_centroids = compute_class_centroids(subset_emb, subset_labels, list(range(k)))
    hybrid_proto    = alpha * class_centroids + (1 - alpha) * label_prototypes[combo]
    return elimination_label_assignment(cluster_centroids, hybrid_proto)


def run(dataset_name, data_path, labeling_mode, max_per_class=None):
    dataset     = load_dataset(dataset_name, data_path)
    texts       = dataset.get_texts()
    labels      = dataset.get_labels()
    class_names = dataset.get_class_names()

    if max_per_class is not None:
        texts, labels = subsample_dataset(texts, labels, max_per_class)
        print(f"Subsampled to max {max_per_class} per class → {len(labels)} total samples")

    # Subsampled runs use a separate cache + results namespace so they never
    # overwrite or load the full-dataset versions.
    cache_tag = f"{dataset_name}_sub{max_per_class}" if max_per_class else dataset_name

    combinations = generate_all_combinations(list(range(len(class_names))))
    alpha_values = [0.25, 0.5, 0.75] if labeling_mode == "hybrid" else [None]
    logger       = ExperimentLogger(output_file(cache_tag, labeling_mode))

    total_computed = total_skipped = 0

    for model_name in ALL_EMBEDDING_MODELS:
        print(f"\nEmbedding: {model_name}")
        embeddings = normalize(get_embedding(model_name, texts, cache_tag))

        label_prototypes = None
        if labeling_mode in ("wikidominer", "hybrid"):
            print(f"  Building WikiDoMiner prototypes...")
            embedder = build_embedder(model_name, texts)
            label_prototypes = build_wikidominer_label_prototypes(
                texts, labels, class_names, embedder, cache_tag, model_name
            )

        for combo in tqdm(combinations):
            k     = len(combo)
            combo = list(combo)
            mask  = np.isin(labels, combo)

            subset_emb    = embeddings[mask]
            subset_labels_raw = labels[mask]
            label_map     = {cls: i for i, cls in enumerate(combo)}
            subset_labels = np.array([label_map[l] for l in subset_labels_raw])

            for method in CLUSTERING_METHODS:
                for alpha in alpha_values:
                    if logger.is_completed(k, combo, model_name, method, alpha):
                        total_skipped += 1
                        continue

                    # --- clustering ---
                    if method in ("dbscan", "hdbscan"):
                        assign_fn    = lambda centroids: get_assignment(
                            labeling_mode, alpha, subset_emb, subset_labels,
                            centroids, label_prototypes, combo, k
                        )
                        cluster_labels = tune_density(method, subset_emb, k, assign_fn)
                        if cluster_labels is None:
                            continue
                    else:
                        cluster_labels = cluster(method, subset_emb, k)

                    unique = set(cluster_labels) - {-1}
                    if len(unique) != k:
                        continue

                    # --- label assignment ---
                    centroids  = compute_cluster_centroids(subset_emb, cluster_labels, k)
                    assignment = get_assignment(labeling_mode, alpha, subset_emb, subset_labels,
                                               centroids, label_prototypes, combo, k)
                    predicted  = map_clusters_to_labels(cluster_labels, assignment)
                    metrics    = compute_macro_metrics(subset_labels, predicted)

                    logger.log(k, combo, model_name, method,
                               metrics["precision"], metrics["recall"], metrics["f1"],
                               alpha=alpha)
                    total_computed += 1

    print(f"\nDone | labeling_mode={labeling_mode} | computed={total_computed} | skipped={total_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",      type=str, required=True)
    parser.add_argument("--path",         type=str, required=True)
    parser.add_argument("--labeling_mode", type=str, default="centroid",
                        choices=["centroid", "wikidominer", "hybrid"])
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Cap each class at N samples before running (e.g. 150)")
    args = parser.parse_args()
    run(args.dataset, args.path, args.labeling_mode, args.max_per_class)
