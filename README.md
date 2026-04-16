# Unsupervised Requirements Classification

An experimental framework for studying **embedding-based unsupervised classification of software requirements**. The project investigates whether modern text embeddings can cluster requirement statements well enough to approximate supervised classification — without any labeled training data.

---

## Research Questions

- How well do modern sentence embeddings capture semantic structure in requirement statements?
- Can unsupervised clustering replace (or approximate) supervised classification for requirements?
- How does clustering performance degrade as the number of requirement categories grows?
- Which embedding model and clustering algorithm work best per dataset?
- Does incorporating external knowledge (WikiDoMiner) improve label assignment?

---

## Repository Structure

```
unsupervised_requirements_01/
│
├── core.py                  All shared utilities (datasets, embeddings,
│                            clustering, labeling, metrics, logger)
│
├── run_unsupervised.py      Run unsupervised experiments
├── run_supervised.py        Run supervised baseline (TF-IDF + Logistic Regression)
├── analysis.py              Post-experiment analysis and plots
├── inspect_data.py          Inspect dataset distributions / preview subsampling
│
├── data/
│   ├── PROMISE_exp.arff     PROMISE dataset  (969 samples, 12 classes)
│   ├── requirements.csv     CrowdRE dataset  (2966 samples, 5 classes)
│   ├── secreq.csv           SecReq dataset   (510 samples, 2 classes)
│   ├── Final.arff           Final dataset    (~625 samples, 10 classes)
│   └── PURE.csv             PURE dataset     (~3000 samples, 3 classes)
│
├── pretrained/              Pre-trained word-embedding files (not in git)
│   ├── glove.6B.300d.txt
│   ├── GoogleNews-vectors-negative300.bin
│   └── wiki-news-300d-1M.vec
│
├── cache/
│   ├── embeddings/          Cached .npy embedding arrays (per dataset)
│   └── wikidominer/         Cached WikiDoMiner prototype arrays
│
├── results/
│   ├── *_unsupervised_centroid_full.csv    Centroid labeling results
│   ├── *_unsupervised_wikidominer_full.csv WikiDoMiner results
│   ├── *_unsupervised_hybrid_full.csv      Hybrid labeling results
│   ├── *_supervised_full.csv            Supervised baseline results
│   ├── analysis/{dataset}/              Centroid analysis outputs
│   ├── analysis_wikidominer/{dataset}/  WikiDoMiner analysis outputs
│   └── analysis_hybrid/{dataset}/       Hybrid analysis outputs
│
└── requirements.txt
```

---

## Pipeline Overview

```
Raw Dataset
    │
    ▼
Dataset Loader  (PromiseDataset / CrowdREDataset / SecReqDataset / FinalDataset / PUREDataset)
    │  parses ARFF or CSV, encodes labels numerically
    │
    ▼
[Optional] Subsampling  (--max_per_class N)
    │  caps each class at N samples for faster experimentation
    │
    ▼
Text Cleaning  (TextCleaner)
    │  lowercase → remove digits/punctuation → stopword removal → lemmatize
    │
    ▼
Embedding Generation  (12 models)
    │  result cached as .npy in cache/embeddings/{dataset}/
    │
    ▼
L2 Normalisation
    │
    ▼
All k-class Subset Combinations  (k = 2 … n_classes)
    │
    ▼
Clustering  (KMeans / HAC / Spectral / DBSCAN / HDBSCAN)
    │  density-based methods grid-search over their hyperparameters
    │
    ▼
Label Assignment
    │  centroid  — cosine similarity + elimination between cluster and class centroids
    │  wikidominer — Wikipedia-derived label prototypes (cached)
    │  hybrid    — alpha * centroid_proto + (1-alpha) * wiki_proto
    │
    ▼
Evaluation  (Macro Precision / Recall / F1)
    │
    ▼
CSV Logging  (resume-safe: already-completed rows are skipped on re-run)
```

---

## Datasets

| Key | File | Samples | Classes | Notes |
|-----|------|---------|---------|-------|
| `promise` | `data/PROMISE_exp.arff` | 969 | 12 | NFR taxonomy (ARFF) |
| `crowdre` | `data/requirements.csv` | 2966 | 5 | Crowd-sourced requirements (CSV) |
| `secreq` | `data/secreq.csv` | 510 | 2 | Security vs. non-security (CSV) |
| `final` | `data/Final.arff` | ~625 | 10 | Multi-category NFRs; ARFF with quoted fields |
| `pure` | `data/PURE.csv` | ~3000 | 3 | Binary security/reliability flags → Functional / Reliability / Security |

**PURE label derivation:** `security=1 → Security`; `reliability=1 AND security=0 → Reliability`; else `Functional`.

---

## Embedding Models

| Key | Model |
|-----|-------|
| `sbert` | paraphrase-MiniLM-L6-v2 |
| `sroberta` | all-distilroberta-v1 |
| `mpnet` | all-mpnet-base-v2 |
| `minilm` | all-MiniLM-L12-v2 |
| `e5` | intfloat/e5-base-v2 |
| `instructor` | hkunlp/instructor-base |
| `bge` | BAAI/bge-base-en |
| `gte` | thenlper/gte-base |
| `w2v_self` | Word2Vec trained on the dataset itself |
| `w2v_pretrained` | GoogleNews Word2Vec (300d) |
| `glove_pretrained` | GloVe 6B 300d |
| `fasttext_pretrained` | FastText wiki-news 300d |

---

## Clustering Methods

| Method | Notes |
|--------|-------|
| `kmeans` | k-means, n_init=10, random_state=42 |
| `hac` | Hierarchical Agglomerative, Ward linkage |
| `spectral` | Spectral clustering, KNN affinity |
| `dbscan` | Grid-searched over eps ∈ {0.3,0.5,0.7,1.0}, min_samples ∈ {3,5,10} |
| `hdbscan` | Grid-searched over min_cluster_size ∈ {3,5,10,15} |

---

## Labeling Modes

| Mode | How clusters get assigned to class labels |
|------|------------------------------------------|
| `centroid` | Cluster centroids matched to true class centroids via cosine similarity + greedy elimination |
| `wikidominer` | Top keywords per class → Wikipedia articles → embed → prototype vectors |
| `hybrid` | `alpha × centroid_proto + (1-alpha) × wiki_proto`, alpha ∈ {0.25, 0.5, 0.75} |

---

## Quickstart

### 1. Inspect datasets

```bash
python inspect_data.py                          # all five datasets
python inspect_data.py --dataset promise        # just PROMISE
python inspect_data.py --dataset final          # just Final
python inspect_data.py --dataset pure           # just PURE
python inspect_data.py --dataset promise --max_per_class 150   # preview subsampling
```

### 2. Run unsupervised experiments

```bash
# Full data — original datasets
python run_unsupervised.py --dataset promise  --path data/PROMISE_exp.arff
python run_unsupervised.py --dataset crowdre  --path data/requirements.csv
python run_unsupervised.py --dataset secreq   --path data/secreq.csv

# Full data — new datasets
python run_unsupervised.py --dataset final    --path data/Final.arff
python run_unsupervised.py --dataset pure     --path data/PURE.csv

# WikiDoMiner labeling
python run_unsupervised.py --dataset promise  --path data/PROMISE_exp.arff --labeling_mode wikidominer
python run_unsupervised.py --dataset final    --path data/Final.arff        --labeling_mode wikidominer
python run_unsupervised.py --dataset pure     --path data/PURE.csv          --labeling_mode wikidominer

# Hybrid labeling
python run_unsupervised.py --dataset promise  --path data/PROMISE_exp.arff --labeling_mode hybrid
python run_unsupervised.py --dataset final    --path data/Final.arff        --labeling_mode hybrid
python run_unsupervised.py --dataset pure     --path data/PURE.csv          --labeling_mode hybrid

# Subsampled (faster — recommended for PROMISE and Final which are compute-heavy)
python run_unsupervised.py --dataset promise  --path data/PROMISE_exp.arff --max_per_class 150
python run_unsupervised.py --dataset final    --path data/Final.arff        --max_per_class 150
```

### 3. Run supervised baseline

```bash
python run_supervised.py --dataset promise  --path data/PROMISE_exp.arff
python run_supervised.py --dataset crowdre  --path data/requirements.csv
python run_supervised.py --dataset secreq   --path data/secreq.csv
python run_supervised.py --dataset final    --path data/Final.arff
python run_supervised.py --dataset pure     --path data/PURE.csv

# Subsampled version (must match the unsupervised cap for fair comparison)
python run_supervised.py --dataset promise  --path data/PROMISE_exp.arff --max_per_class 150
python run_supervised.py --dataset final    --path data/Final.arff        --max_per_class 150
```

### 4. Run analysis

```bash
# Centroid labeling analysis
python analysis.py --dataset promise
python analysis.py --dataset crowdre
python analysis.py --dataset secreq
python analysis.py --dataset final
python analysis.py --dataset pure

# WikiDoMiner analysis
python analysis.py --dataset promise --labeling_mode wikidominer
python analysis.py --dataset crowdre --labeling_mode wikidominer
python analysis.py --dataset secreq  --labeling_mode wikidominer
python analysis.py --dataset final   --labeling_mode wikidominer
python analysis.py --dataset pure    --labeling_mode wikidominer

# Hybrid analysis
python analysis.py --dataset promise --labeling_mode hybrid
python analysis.py --dataset crowdre --labeling_mode hybrid
python analysis.py --dataset secreq  --labeling_mode hybrid
python analysis.py --dataset final   --labeling_mode hybrid
python analysis.py --dataset pure    --labeling_mode hybrid

# Subsampled analysis (pass same cap used during experiment)
python analysis.py --dataset promise --max_per_class 150
python analysis.py --dataset final   --max_per_class 150
```

---

## Analysis Outputs

Each `analysis.py` run writes the following to `results/analysis[_mode]/{dataset}/`:

| File | Description |
|------|-------------|
| `summary_stats.csv` | Mean ± std F1 grouped by embedding × clustering × k |
| `degradation_kmeans.png` | F1 vs k per embedding (KMeans only) |
| `degradation_comparison.png` | Unsupervised vs supervised F1 degradation |
| `degradation_slopes.csv` | Linear regression slope of F1 vs k per method |
| `performance_gap.csv` | Supervised − unsupervised F1 gap per k |
| `robustness_ratio.csv` | F1(k=max) / F1(k=2) — how much performance drops at scale |
| `clustering_comparison.csv` | KMeans vs HAC F1 difference per embedding × k |
| `binary_class_difficulty.csv` | Average F1 per class in binary (k=2) classification |
| `ranking_per_k.csv` | Method ranking by mean F1 at each k |
| `alpha_analysis.csv` | *(hybrid only)* Mean F1 per alpha value |
| `alpha_plot.png` | *(hybrid only)* Alpha vs average F1 |

---

## Subsampling for Constrained Compute

PROMISE has 12 classes with severe imbalance (F=444 samples vs PO=12). The Final dataset also has 10 classes with class imbalance. Running all combinations of many classes × 12 embeddings × 5 clustering methods is very expensive.

**Recommended approach:**
```bash
python inspect_data.py --dataset promise --max_per_class 150
# → shows 675 total samples; only F is capped; all other classes kept in full

python run_unsupervised.py --dataset promise --path data/PROMISE_exp.arff --max_per_class 150
python run_supervised.py   --dataset promise --path data/PROMISE_exp.arff --max_per_class 150
python analysis.py         --dataset promise --max_per_class 150

# Same pattern for Final
python run_unsupervised.py --dataset final --path data/Final.arff --max_per_class 150
python run_supervised.py   --dataset final --path data/Final.arff --max_per_class 150
python analysis.py         --dataset final --max_per_class 150
```

Results are stored under a separate namespace (`promise_sub150_*`) so they never overwrite full-dataset results. Embeddings are cached separately too.

---

## Resume Safety

All experiment runners are resume-safe. If a run is interrupted, re-running the same command skips already-logged rows and continues from where it left off. This is handled by `ExperimentLogger` in `core.py` which maintains an in-memory set of completed `(k, subset, embedding, clustering, alpha)` tuples loaded from the output CSV at startup.

---

## Notes

- Pretrained word-embedding files are **not included in the repo** (too large). Download them separately and place in `pretrained/`.
- GloVe `.txt` is auto-converted to word2vec format on first use (saved as `.txt.word2vec`).
- Embedding arrays are cached as `.npy` files in `cache/` — delete them to force recomputation.
- WikiDoMiner prototypes are cached per dataset × embedding in `cache/wikidominer/`.
