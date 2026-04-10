# =============================================================================
# core.py — All shared utilities for the unsupervised requirements pipeline
# =============================================================================
# Covers: datasets, text cleaning, embeddings (transformer + word), clustering,
#         labeling (centroid + wikidominer), metrics, experiment logger,
#         and combination generator.
# =============================================================================

import os
import re
import csv
import ast
import itertools
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# DATASETS
# =============================================================================

class PromiseDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.texts, self.labels, self.class_names = [], [], None
        self.label_encoder, self.label_decoder = {}, {}

    def load(self):
        in_data = False
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.lower() == "@data":
                    in_data = True
                    continue
                if not in_data:
                    continue
                first, last = line.find(","), line.rfind(",")
                text = line[first + 1:last].strip().strip("'")
                label = line[last + 1:].strip()
                self.texts.append(text)
                self.labels.append(label)

        self.class_names = sorted(set(self.labels))
        self.label_encoder = {l: i for i, l in enumerate(self.class_names)}
        self.label_decoder = {i: l for l, i in self.label_encoder.items()}
        self.labels = np.array([self.label_encoder[l] for l in self.labels])
        return self

    def get_texts(self):      return self.texts
    def get_labels(self):     return self.labels
    def get_class_names(self): return self.class_names


class CrowdREDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.texts, self.labels, self.class_names = [], [], None
        self.label_encoder, self.label_decoder = {}, {}

    def load(self):
        df = pd.read_csv(self.file_path)[["feature", "benefit", "application_domain"]].dropna()
        df["text"] = df["feature"].astype(str) + " " + df["benefit"].astype(str)
        raw_labels = df["application_domain"].tolist()
        self.class_names = sorted(set(raw_labels))
        self.label_encoder = {l: i for i, l in enumerate(self.class_names)}
        self.label_decoder = {i: l for l, i in self.label_encoder.items()}
        self.labels = np.array([self.label_encoder[l] for l in raw_labels])
        self.texts = df["text"].tolist()
        return self

    def get_texts(self):      return self.texts
    def get_labels(self):     return self.labels
    def get_class_names(self): return self.class_names


class SecReqDataset:
    def __init__(self, file_path):
        self.file_path = file_path
        self.texts, self.labels, self.class_names = [], [], None
        self.label_encoder, self.label_decoder = {}, {}

    def load(self):
        df = pd.read_csv(self.file_path)
        self.texts = df["text"].astype(str).tolist()
        raw_labels = df["label"].astype(str).tolist()
        self.class_names = sorted(set(raw_labels))
        self.label_encoder = {l: i for i, l in enumerate(self.class_names)}
        self.label_decoder = {i: l for l, i in self.label_encoder.items()}
        self.labels = np.array([self.label_encoder[l] for l in raw_labels])
        return self

    def get_texts(self):      return self.texts
    def get_labels(self):     return self.labels
    def get_class_names(self): return self.class_names


def load_dataset(dataset_name, path):
    loaders = {
        "promise": PromiseDataset,
        "crowdre": CrowdREDataset,
        "secreq":  SecReqDataset,
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return loaders[dataset_name](path).load()


# =============================================================================
# TEXT CLEANING
# =============================================================================

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("omw-1.4",   quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextCleaner:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text):
        text = text.lower()
        text = re.sub(r"\d+", "", text)
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = [self.lemmatizer.lemmatize(t) for t in text.split() if t not in self.stop_words]
        return " ".join(tokens)


# =============================================================================
# EMBEDDINGS
# =============================================================================

TRANSFORMER_MODELS = {
    "sbert":     "paraphrase-MiniLM-L6-v2",
    "sroberta":  "all-distilroberta-v1",
    "mpnet":     "all-mpnet-base-v2",
    "minilm":    "sentence-transformers/all-MiniLM-L12-v2",
    "e5":        "intfloat/e5-base-v2",
    "instructor":"hkunlp/instructor-base",
    "bge":       "BAAI/bge-base-en",
    "gte":       "thenlper/gte-base",
}

WORD_PRETRAINED_MODELS = {
    "w2v_pretrained":    "pretrained/GoogleNews-vectors-negative300.bin",
    "fasttext_pretrained":"pretrained/wiki-news-300d-1M.vec",
    "glove_pretrained":  "pretrained/glove.6B.300d.txt",
}

ALL_EMBEDDING_MODELS = list(TRANSFORMER_MODELS) + ["w2v_self"] + list(WORD_PRETRAINED_MODELS)


class TransformerEmbedder:
    def __init__(self, model_name):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return np.array(self.model.encode(texts, batch_size=32, show_progress_bar=True))


class WordEmbedder:
    def __init__(self, mode, model_path=None, texts=None, vector_size=300):
        self.mode = mode

        if mode == "pretrained":
            from gensim.models import KeyedVectors
            if model_path.endswith(".txt"):
                converted = model_path + ".word2vec"
                if not os.path.exists(converted):
                    from gensim.scripts.glove2word2vec import glove2word2vec
                    print("Converting GloVe to word2vec format...")
                    glove2word2vec(model_path, converted)
                model_path, binary = converted, False
            elif model_path.endswith(".vec"):
                binary = False
            else:
                binary = True
            self.model = KeyedVectors.load_word2vec_format(model_path, binary=binary)
            self.vector_size = self.model.vector_size

        elif mode == "self":
            from gensim.models import Word2Vec
            tokenized = [t.split() for t in texts]
            w2v = Word2Vec(sentences=tokenized, vector_size=vector_size, window=5, min_count=1, workers=4)
            self.model = w2v.wv
            self.vector_size = vector_size

        else:
            raise ValueError("Unknown word embedding mode")

    def encode(self, texts):
        embeddings = []
        for text in texts:
            vecs = [self.model[t] for t in text.split() if t in self.model]
            embeddings.append(np.mean(vecs, axis=0) if vecs else np.zeros(self.vector_size))
        return np.array(embeddings)


def build_embedder(model_name, texts, cleaner=None):
    """Return an embedder object (used when you need to embed arbitrary text later, e.g. WikiDoMiner)."""
    if cleaner is None:
        cleaner = TextCleaner()
    texts_clean = [cleaner.clean(t) for t in texts]

    if model_name in TRANSFORMER_MODELS:
        return TransformerEmbedder(TRANSFORMER_MODELS[model_name])
    elif model_name == "w2v_self":
        return WordEmbedder(mode="self", texts=texts_clean)
    elif model_name in WORD_PRETRAINED_MODELS:
        return WordEmbedder(mode="pretrained", model_path=WORD_PRETRAINED_MODELS[model_name])
    else:
        raise ValueError(f"Unsupported embedding model: {model_name}")


def get_embedding(model_name, texts, dataset_name):
    """Load from cache if available, otherwise compute and cache."""
    cache_dir = f"cache/embeddings/{dataset_name}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{model_name}.npy")

    if os.path.exists(cache_path):
        print(f"Loaded cached embedding: {model_name}")
        return np.load(cache_path)

    cleaner = TextCleaner()
    texts_clean = [cleaner.clean(t) for t in texts]
    embedder = build_embedder(model_name, texts, cleaner)
    embeddings = embedder.encode(texts_clean)
    np.save(cache_path, embeddings)
    return embeddings


# =============================================================================
# CLUSTERING
# =============================================================================

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, DBSCAN
try:
    import hdbscan as _hdbscan
except ImportError:
    _hdbscan = None


def run_kmeans(embeddings, k):
    return KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(embeddings)

def run_hac(embeddings, k):
    return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(embeddings)

def run_spectral(embeddings, k):
    return SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                              assign_labels="kmeans", random_state=42).fit_predict(embeddings)

def run_dbscan(embeddings, eps=0.5, min_samples=5):
    return DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit_predict(embeddings)

def run_hdbscan(embeddings, min_cluster_size=5):
    if _hdbscan is None:
        raise ImportError("Install hdbscan: pip install hdbscan")
    return _hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean").fit_predict(embeddings)

def compute_cluster_centroids(embeddings, cluster_labels, n_clusters):
    centroids = []
    for i in range(n_clusters):
        pts = embeddings[cluster_labels == i]
        centroids.append(np.mean(pts, axis=0) if len(pts) else np.zeros(embeddings.shape[1]))
    return np.array(centroids)


# =============================================================================
# LABELING — CENTROID-BASED
# =============================================================================

def compute_class_centroids(embeddings, labels, class_subset):
    return np.array([np.mean(embeddings[labels == c], axis=0) for c in class_subset])

def elimination_label_assignment(cluster_centroids, class_centroids):
    sim = cosine_similarity(cluster_centroids, class_centroids)
    n = sim.shape[0]
    assigned = [-1] * n
    for _ in range(n):
        ci, cj = np.unravel_index(np.argmax(sim), sim.shape)
        assigned[ci] = cj
        sim[ci, :] = -1
        sim[:, cj] = -1
    return assigned

def map_clusters_to_labels(cluster_labels, assignment):
    predicted = np.zeros_like(cluster_labels)
    for cluster_id, class_id in enumerate(assignment):
        predicted[cluster_labels == cluster_id] = class_id
    return predicted


# =============================================================================
# LABELING — WIKIDOMINER
# =============================================================================

def _wiki_clean(text):
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def extract_keywords(texts, top_k=10):
    words = []
    for t in texts:
        words.extend(_wiki_clean(t).split())
    freq = {w: c for w, c in Counter(words).items() if len(w) > 2}
    return [w for w, _ in Counter(freq).most_common(top_k)]

def fetch_wikipedia_articles(keywords, max_per_keyword=2):
    import wikipedia
    corpus = []
    for kw in keywords:
        try:
            for title in wikipedia.search(kw)[:max_per_keyword]:
                try:
                    corpus.append(wikipedia.page(title, auto_suggest=False).content)
                except Exception:
                    pass
        except Exception:
            pass
    return corpus

def build_wikidominer_label_prototypes(texts, labels, class_names, embedder,
                                        dataset_name, embedding_name, top_k=10):
    cache_dir = f"cache/wikidominer/{dataset_name}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{embedding_name}.npy")

    if os.path.exists(cache_path):
        print(f"Loaded WikiDoMiner cache: {cache_path}")
        return np.load(cache_path)

    print(f"Building WikiDoMiner prototypes for {embedding_name}...")
    label_prototypes = []

    for class_id, class_name in enumerate(class_names):
        print(f"  Building corpus for class: {class_name}")
        class_texts = [texts[i] for i in range(len(texts)) if labels[i] == class_id]

        if not class_texts:
            dim = embedder.encode(["test"]).shape[1]
            label_prototypes.append(np.zeros(dim))
            continue

        wiki_corpus = fetch_wikipedia_articles(extract_keywords(class_texts, top_k)) or class_texts
        prototype = np.mean(embedder.encode(wiki_corpus), axis=0)
        label_prototypes.append(prototype)

    label_prototypes = np.array(label_prototypes)
    np.save(cache_path, label_prototypes)
    print(f"Saved WikiDoMiner cache: {cache_path}")
    return label_prototypes

def assign_clusters_wikidominer(cluster_centroids, label_prototypes):
    sim = cosine_similarity(cluster_centroids, label_prototypes)
    n = sim.shape[0]
    assigned = [-1] * n
    for _ in range(n):
        i, j = np.unravel_index(np.argmax(sim), sim.shape)
        assigned[i] = j
        sim[i, :] = -1
        sim[:, j] = -1
    return assigned


# =============================================================================
# METRICS
# =============================================================================

def compute_macro_metrics(true_labels, predicted_labels):
    p, r, f1, _ = precision_recall_fscore_support(
        true_labels, predicted_labels, average="macro", zero_division=0
    )
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


# =============================================================================
# EXPERIMENT LOGGER
# =============================================================================

class ExperimentLogger:
    def __init__(self, file_path):
        self.file_path = file_path
        self.completed = set()
        if os.path.exists(file_path):
            self._load_completed()
        else:
            self._create_file()

    def _create_file(self):
        os.makedirs(os.path.dirname(self.file_path) or ".", exist_ok=True)
        with open(self.file_path, "w", newline="") as f:
            csv.writer(f).writerow(["k", "class_subset", "embedding", "clustering",
                                    "precision", "recall", "f1", "alpha"])

    def _load_completed(self):
        with open(self.file_path, "r", newline="") as f:
            for row in csv.DictReader(f):
                alpha = row.get("alpha")
                alpha = float(alpha) if alpha and alpha != "" else None
                key = (int(row["k"]), tuple(sorted(ast.literal_eval(row["class_subset"]))),
                       row["embedding"], row["clustering"], alpha)
                self.completed.add(key)

    def _key(self, k, subset, embedding, clustering, alpha):
        return (k, tuple(sorted(subset)), embedding, clustering, alpha)

    def is_completed(self, k, subset, embedding, clustering, alpha=None):
        return self._key(k, subset, embedding, clustering, alpha) in self.completed

    def log(self, k, subset, embedding, clustering, precision, recall, f1, alpha=None):
        key = self._key(k, subset, embedding, clustering, alpha)
        if key in self.completed:
            return
        with open(self.file_path, "a", newline="") as f:
            csv.writer(f).writerow([k, list(tuple(sorted(subset))), embedding, clustering,
                                    precision, recall, f1, alpha])
        self.completed.add(key)


# =============================================================================
# SUBSAMPLING
# =============================================================================

def subsample_dataset(texts, labels, max_per_class, random_state=42):
    """
    Return (texts_sub, labels_sub) keeping at most `max_per_class` samples per
    class.  Classes with fewer samples are kept in full.  Order is shuffled
    within each class for reproducibility via `random_state`.
    """
    import random
    rng = random.Random(random_state)

    indices_by_class = {}
    for i, lbl in enumerate(labels):
        indices_by_class.setdefault(int(lbl), []).append(i)

    kept = []
    for cls_indices in indices_by_class.values():
        shuffled = cls_indices[:]
        rng.shuffle(shuffled)
        kept.extend(shuffled[:max_per_class])

    kept.sort()  # preserve original ordering across classes
    texts_sub  = [texts[i] for i in kept]
    labels_sub = labels[kept]
    return texts_sub, labels_sub


# =============================================================================
# COMBINATION GENERATOR
# =============================================================================

def generate_all_combinations(class_indices):
    """All k-subsets for k in [2, n]."""
    combos = []
    for k in range(2, len(class_indices) + 1):
        combos.extend(itertools.combinations(class_indices, k))
    return combos
