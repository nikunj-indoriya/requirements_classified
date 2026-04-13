# =============================================================================
# inspect_data.py — Dataset distribution inspector
# =============================================================================
# Usage:
#   python inspect_data.py                        # all three datasets
#   python inspect_data.py --dataset promise
#   python inspect_data.py --dataset promise --max_per_class 150
# =============================================================================

import argparse
from collections import Counter
from core import load_dataset

DATASETS = {
    "promise": "data/PROMISE_exp.arff",
    "crowdre": "data/requirements.csv",
    "secreq":  "data/secreq.csv",
    "final":   "data/Final.arff",
    "pure":    "data/PURE.csv",
}


def inspect(dataset_name, path, max_per_class=None):
    dataset     = load_dataset(dataset_name, path)
    texts       = dataset.get_texts()
    labels      = dataset.get_labels()
    class_names = dataset.get_class_names()
    counts      = Counter(labels)

    total   = len(labels)
    n_class = len(class_names)
    min_cnt = min(counts.values())
    max_cnt = max(counts.values())
    avg_cnt = total / n_class

    print(f"\n{'='*55}")
    print(f"  Dataset : {dataset_name.upper()}")
    print(f"  Samples : {total}   Classes: {n_class}")
    print(f"  Min/Avg/Max per class: {min_cnt} / {avg_cnt:.1f} / {max_cnt}")
    print(f"{'='*55}")
    print(f"  {'#':<4} {'Class':<30} {'Count':>6}  {'Bar'}")
    print(f"  {'-'*52}")

    bar_scale = 40 / max_cnt
    for idx, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        bar  = "█" * int(cnt * bar_scale)
        print(f"  [{idx:>2}] {class_names[idx]:<30} {cnt:>5}")

    if max_per_class is not None:
        capped_total = sum(min(c, max_per_class) for c in counts.values())
        kept_classes = sum(1 for c in counts.values() if c >= max_per_class)
        print(f"\n  Subsampling preview  (max_per_class={max_per_class})")
        print(f"  {'#':<4} {'Class':<30} {'Original':>8} → {'Kept':>6}")
        print(f"  {'-'*52}")
        for idx, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            kept = min(cnt, max_per_class)
            marker = "  (capped)" if cnt > max_per_class else ""
            print(f"  [{idx:>2}] {class_names[idx]:<30} {cnt:>7}  → {kept:>5}{marker}")
        print(f"\n  Total after subsampling: {capped_total}  "
              f"(classes fully capped: {kept_classes}/{n_class})")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",       type=str, default=None,
                        choices=list(DATASETS), help="Specific dataset (default: all)")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Preview subsampling at this cap")
    args = parser.parse_args()

    targets = {args.dataset: DATASETS[args.dataset]} if args.dataset else DATASETS
    for name, path in targets.items():
        inspect(name, path, args.max_per_class)


if __name__ == "__main__":
    main()
