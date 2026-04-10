# =============================================================================
# run_supervised.py — Supervised baseline (TF-IDF + Logistic Regression)
# =============================================================================
# Usage:
#   python run_supervised.py --dataset promise --path data/PROMISE_exp.arff
#   python run_supervised.py --dataset crowdre --path data/requirements.csv
#   python run_supervised.py --dataset secreq  --path data/secreq.csv
# =============================================================================

import argparse
import numpy as np
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import precision_recall_fscore_support

from core import load_dataset, generate_all_combinations, ExperimentLogger, subsample_dataset


def compute_macro_metrics(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return p, r, f1


def run(dataset_name, data_path, max_per_class=None):
    dataset  = load_dataset(dataset_name, data_path)
    texts    = np.array(dataset.get_texts())
    labels   = np.array(dataset.get_labels())
    combos   = generate_all_combinations(list(range(len(dataset.get_class_names()))))

    if max_per_class is not None:
        texts, labels = subsample_dataset(list(texts), labels, max_per_class)
        texts = np.array(texts)
        print(f"Subsampled to max {max_per_class} per class → {len(labels)} total samples")

    cache_tag = f"{dataset_name}_sub{max_per_class}" if max_per_class else dataset_name
    logger    = ExperimentLogger(f"results/{cache_tag}_supervised_full.csv")
    total_computed = total_skipped = 0

    for combo in tqdm(combos):
        k, combo = len(combo), list(combo)

        if logger.is_completed(k, combo, "logistic_regression", "supervised"):
            total_skipped += 1
            continue

        mask          = np.isin(labels, combo)
        subset_texts  = texts[mask]
        subset_labels = np.array([{cls: i for i, cls in enumerate(combo)}[l] for l in labels[mask]])

        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        fold_p, fold_r, fold_f = [], [], []

        for train_idx, test_idx in skf.split(subset_texts, subset_labels):
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
            X_tr = vec.fit_transform(subset_texts[train_idx])
            X_te = vec.transform(subset_texts[test_idx])
            y_tr, y_te = subset_labels[train_idx], subset_labels[test_idx]

            clf  = LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs")
            grid = GridSearchCV(clf, {"C": [0.01, 0.1, 1, 10]}, cv=3, scoring="f1_macro", n_jobs=-1)
            grid.fit(X_tr, y_tr)

            p, r, f1 = compute_macro_metrics(y_te, grid.best_estimator_.predict(X_te))
            fold_p.append(p); fold_r.append(r); fold_f.append(f1)

        logger.log(k, combo, "logistic_regression", "supervised",
                   round(np.mean(fold_p), 4), round(np.mean(fold_r), 4), round(np.mean(fold_f), 4))
        total_computed += 1

    print(f"\nDone | computed={total_computed} | skipped={total_skipped}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",       type=str, required=True)
    parser.add_argument("--path",          type=str, required=True)
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Cap each class at N samples before running (e.g. 150)")
    args = parser.parse_args()
    run(args.dataset, args.path, args.max_per_class)
