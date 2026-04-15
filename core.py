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


class FinalDataset:
    """
    Final.arff — 13-attribute ARFF with multi-category requirement classification.
    Text  : attribute 0  (Requirements)
    Label : attribute 2  (Requirement Category)
    Valid labels: Functional, Usability, Reliability & Availability, Performance,
                  Security, Supportability, Constraints, Interfaces, Standards, Safety
    Rows whose label field does not match a valid category are dropped (bad parses).
    File uses latin-1 encoding and single-quote-delimited strings in the data section.
    """
    _VALID_LABELS = {
        "Functional", "Usability", "Reliability & Availability", "Performance",
        "Security", "Supportability", "Constraints", "Interfaces", "Standards", "Safety",
    }

    def __init__(self, file_path):
        self.file_path = file_path
        self.texts, self.labels, self.class_names = [], [], None
        self.label_encoder, self.label_decoder = {}, {}

    def load(self):
        import csv as _csv, io as _io
        in_data = False
        raw_texts, raw_labels = [], []

        with open(self.file_path, "rb") as f:
            for line_bytes in f:
                try:
                    line = line_bytes.decode("latin-1").strip()
                except UnicodeDecodeError:
                    continue
                if not line:
                    continue
                if line.lower() == "@data":
                    in_data = True
                    continue
                if not in_data:
                    continue
                try:
                    reader = _csv.reader(_io.StringIO(line), quotechar="'",
                                         skipinitialspace=True)
                    vals = next(reader)
                    if len(vals) < 3:
                        continue
                    label = vals[2].strip()
                    if label not in self._VALID_LABELS:
                        continue
                    raw_texts.append(vals[0].strip())
                    raw_labels.append(label)
                except Exception:
                    continue

        self.class_names = sorted(self._VALID_LABELS)
        self.label_encoder = {l: i for i, l in enumerate(self.class_names)}
        self.label_decoder = {i: l for l, i in self.label_encoder.items()}
        self.texts  = raw_texts
        self.labels = np.array([self.label_encoder[l] for l in raw_labels])
        return self

    def get_texts(self):       return self.texts
    def get_labels(self):      return self.labels
    def get_class_names(self): return self.class_names


class PUREDataset:
    """
    PURE.csv — public requirements documents dataset.
    Text  : 'sentence' column
    Labels derived from binary flag columns:
        security=1              → 'Security'
        reliability=1, sec=0   → 'Reliability'
        both 0                  → 'Functional'
    Duplicate sentences are dropped (keep first occurrence).
    File uses latin-1 encoding.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.texts, self.labels, self.class_names = [], [], None
        self.label_encoder, self.label_decoder = {}, {}

    def load(self):
        df = pd.read_csv(self.file_path, encoding="latin-1")
        df = df.dropna(subset=["sentence"])
        df = df.drop_duplicates(subset=["sentence"])

        def _derive_label(row):
            if row["security"] == 1:
                return "Security"
            if row["reliability"] == 1:
                return "Reliability"
            return "Functional"

        raw_labels = df.apply(_derive_label, axis=1).tolist()
        self.class_names = sorted(set(raw_labels))
        self.label_encoder = {l: i for i, l in enumerate(self.class_names)}
        self.label_decoder = {i: l for l, i in self.label_encoder.items()}
        self.texts  = df["sentence"].astype(str).tolist()
        self.labels = np.array([self.label_encoder[l] for l in raw_labels])
        return self

    def get_texts(self):       return self.texts
    def get_labels(self):      return self.labels
    def get_class_names(self): return self.class_names


def load_dataset(dataset_name, path):
    loaders = {
        "promise": PromiseDataset,
        "crowdre": CrowdREDataset,
        "secreq":  SecReqDataset,
        "final":   FinalDataset,
        "pure":    PUREDataset,
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
    # n_neighbors must be < n_samples; sklearn default is 10 which fails on small subsets
    n_neighbors = min(10, len(embeddings) - 1)
    if n_neighbors < k:
        raise ValueError(
            f"SpectralClustering: n_neighbors ({n_neighbors}) < k ({k}); subset too small"
        )
    return SpectralClustering(n_clusters=k, affinity="nearest_neighbors",
                              n_neighbors=n_neighbors,
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

def _has_title_overlap(keyword, title):
    """True if keyword and Wikipedia title share at least one non-trivial word."""
    skip = {"the", "a", "an", "of", "in", "for", "and", "or", "to", "is", "are", "was"}
    kw_words   = {w for w in keyword.lower().split() if w not in skip and len(w) > 2}
    title_words = {w for w in title.lower().split()   if w not in skip and len(w) > 2}
    return bool(kw_words & title_words)


def extract_keywords(texts, top_k=50):
    """
    Paper-faithful keyword extraction (Ezzini et al., WikiDoMiner, ESEC/FSE 2022).

    Steps:
      A. spaCy NP chunker  — extract lemmatized noun phrases per document
      B. WordNet filter    — discard NPs that are plain English (in WordNet);
                             multi-word phrases are kept unless their compound
                             form is also in WordNet
      C. TF-IDF ranking   — score each surviving NP across documents, return top-K
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.corpus import wordnet

    # ---- A: NP extraction via spaCy ----------------------------------------
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess, sys
            print("  Downloading spaCy model en_core_web_sm …")
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                           check=True, capture_output=True)
            nlp = spacy.load("en_core_web_sm")
        spacy_ok = True
    except ImportError:
        spacy_ok = False

    if not spacy_ok:
        # Graceful fallback to simple word-frequency (no spaCy)
        words = []
        for t in texts:
            t = re.sub(r"[^\w\s]", " ", t.lower())
            words.extend(w for w in t.split() if len(w) > 2)
        freq = Counter(words)
        return [w for w, _ in freq.most_common(top_k)]

    doc_np_lists = []
    for text in texts:
        doc = nlp(text[:8000])          # cap length to keep speed reasonable
        nps = []
        for chunk in doc.noun_chunks:
            # lemmatise each content word inside the NP
            tokens = [
                token.lemma_.lower()
                for token in chunk
                if not token.is_stop and token.is_alpha and len(token.lemma_) > 1
            ]
            lemma_np = " ".join(tokens).strip()
            if len(lemma_np) > 2:
                nps.append(lemma_np)
        doc_np_lists.append(nps)

    all_nps = set(np_s for nps in doc_np_lists for np_s in nps)

    # ---- B: WordNet filter --------------------------------------------------
    domain_nps = []
    for np_str in all_nps:
        tokens = np_str.strip().split()
        if not tokens:
            continue
        if len(tokens) == 1:
            # Single word: discard if it exists as a generic English word in WordNet
            if not wordnet.synsets(tokens[0]):
                domain_nps.append(np_str)
        else:
            # Multi-word phrase: discard only if the compound itself is in WordNet
            # (e.g. "software_system" is not; "dining_room" is)
            if not wordnet.synsets("_".join(tokens)):
                domain_nps.append(np_str)

    # If filtering is too aggressive, fall back to all NPs
    if len(domain_nps) < min(5, len(all_nps)):
        domain_nps = list(all_nps)

    # ---- C: TF-IDF ranking --------------------------------------------------
    domain_np_set = set(domain_nps)
    # Build per-document NP bags; replace spaces in NPs with underscores so
    # TfidfVectorizer treats each NP as a single token
    doc_bags = []
    for nps in doc_np_lists:
        filtered = [np_s.replace(" ", "_") for np_s in nps if np_s in domain_np_set]
        doc_bags.append(" ".join(filtered) if filtered else "")

    non_empty = [d for d in doc_bags if d.strip()]
    if not non_empty:
        freq = Counter(np_s for nps in doc_np_lists for np_s in nps)
        return [kw for kw, _ in freq.most_common(top_k)]

    try:
        vectorizer   = TfidfVectorizer(ngram_range=(1, 1), min_df=1)
        tfidf_matrix = vectorizer.fit_transform(non_empty)
        scores       = np.asarray(tfidf_matrix.sum(axis=0)).flatten()
        feature_names = vectorizer.get_feature_names_out()
        top_indices  = np.argsort(scores)[::-1][:top_k]
        return [feature_names[i].replace("_", " ") for i in top_indices]
    except Exception:
        freq = Counter(np_s for nps in doc_np_lists for np_s in nps if np_s in domain_np_set)
        return [kw for kw, _ in freq.most_common(top_k)]


def fetch_wikipedia_articles(keywords, depth=1, max_articles=300):
    """
    Paper-faithful Wikipedia querying (Ezzini et al., WikiDoMiner, ESEC/FSE 2022).

    For each keyword:
      1. Search Wikipedia and find an article whose title *partially overlaps*
         with the keyword (not just the top search hit).
      2. Expand the corpus by traversing the Wikipedia category graph:
         depth=0  →  direct matching articles only
         depth=1  →  + all articles in the same categories (Figure 2, depth-1)
         depth=2  →  + articles in subcategories one level deeper (depth-2)

    Uses 'wikipedia' for search and 'wikipediaapi' for category traversal.
    Falls back to simple fetching if wikipediaapi is not installed.
    """
    import wikipedia as _wiki

    try:
        import wikipediaapi
        wiki_api = wikipediaapi.Wikipedia(
            user_agent="UnsupervisedReqClassifier/1.0",
            language="en",
        )
        api_ok = True
    except ImportError:
        api_ok = False

    corpus      = []
    seen_titles = set()

    # ---- fallback (no wikipediaapi) ----------------------------------------
    if not api_ok:
        for kw in keywords:
            try:
                for title in _wiki.search(kw)[:2]:
                    if title not in seen_titles:
                        seen_titles.add(title)
                        try:
                            corpus.append(_wiki.page(title, auto_suggest=False).content)
                        except Exception:
                            pass
            except Exception:
                pass
        return corpus

    # ---- helpers -----------------------------------------------------------
    def _add(title):
        if title in seen_titles or len(corpus) >= max_articles:
            return
        seen_titles.add(title)
        page = wiki_api.page(title)
        if page.exists() and page.summary:
            corpus.append(page.summary)

    def _expand(cat_page, cur_depth, max_depth):
        if len(corpus) >= max_articles:
            return
        try:
            for m_title, member in list(cat_page.categorymembers.items())[:60]:
                if len(corpus) >= max_articles:
                    break
                if member.ns == 0:                      # article namespace
                    _add(member.title)
                elif member.ns == 14 and cur_depth < max_depth:  # category
                    _expand(member, cur_depth + 1, max_depth)
        except Exception:
            pass

    # ---- main loop ---------------------------------------------------------
    for kw in keywords:
        if len(corpus) >= max_articles:
            break
        try:
            search_hits = _wiki.search(kw)[:5]
        except Exception:
            continue

        # Prefer a hit whose title overlaps with the keyword (partial match)
        matched = next((t for t in search_hits if _has_title_overlap(kw, t)), None)
        if not matched and search_hits:
            matched = search_hits[0]         # fallback: top result
        if not matched:
            continue

        page = wiki_api.page(matched)
        if not page.exists():
            continue

        _add(page.title)

        # Category traversal (depth >= 1)
        if depth >= 1:
            try:
                for _cat_title, cat_page in list(page.categories.items())[:10]:
                    if len(corpus) >= max_articles:
                        break
                    _expand(cat_page, 1, depth)
            except Exception:
                pass

    return corpus

def _fetch_and_cache_wiki_articles(texts, labels, class_names, dataset_name,
                                    top_k=50, depth=0):
    """
    Fetch Wikipedia article summaries ONCE per (dataset, top_k, depth) and
    cache as JSON.  All 12 embedding models share this cache — Wikipedia is
    only queried once regardless of how many models are run afterwards.
    """
    import json
    cache_dir  = f"cache/wikidominer/{dataset_name}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"articles_topk{top_k}_depth{depth}.json")

    if os.path.exists(cache_path):
        print(f"  Loaded Wikipedia article cache: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    print(f"  Fetching Wikipedia articles (top_k={top_k}, depth={depth}) — one-time cost...")
    articles_per_class = {}

    for class_id, class_name in enumerate(class_names):
        print(f"    [{class_id+1}/{len(class_names)}] class='{class_name}'")
        class_texts = [texts[i] for i in range(len(texts)) if labels[i] == class_id]

        if not class_texts:
            articles_per_class[class_name] = []
            continue

        keywords    = extract_keywords(class_texts, top_k=top_k)
        print(f"      keywords ({len(keywords)}): {keywords[:5]} …")
        wiki_corpus = fetch_wikipedia_articles(keywords, depth=depth)
        print(f"      fetched {len(wiki_corpus)} article summaries")
        articles_per_class[class_name] = wiki_corpus

    with open(cache_path, "w") as f:
        json.dump(articles_per_class, f)
    print(f"  Saved Wikipedia article cache: {cache_path}")
    return articles_per_class


def build_wikidominer_label_prototypes(texts, labels, class_names, embedder,
                                        dataset_name, embedding_name,
                                        top_k=50, wiki_depth=0):
    """
    Build one Wikipedia-derived embedding prototype per class label.

    Wikipedia articles are fetched and cached ONCE (shared across all embedding
    models). Each model only pays the embedding cost, not the network cost.

    Steps:
      1. _fetch_and_cache_wiki_articles  — NP keywords → Wikipedia → JSON cache
      2. For each class, embed the cached article summaries → mean → prototype
      3. Save prototype matrix to .npy cache
    """
    cache_dir  = f"cache/wikidominer/{dataset_name}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{embedding_name}.npy")

    if os.path.exists(cache_path):
        print(f"Loaded WikiDoMiner prototype cache: {cache_path}")
        return np.load(cache_path)

    print(f"Building WikiDoMiner prototypes for {embedding_name}...")

    # Step 1: get articles (instant if already cached)
    articles_per_class = _fetch_and_cache_wiki_articles(
        texts, labels, class_names, dataset_name, top_k=top_k, depth=wiki_depth
    )

    # Step 2: embed per class
    label_prototypes = []
    for class_id, class_name in enumerate(class_names):
        class_texts  = [texts[i] for i in range(len(texts)) if labels[i] == class_id]
        wiki_articles = articles_per_class.get(class_name, [])
        source_texts  = wiki_articles if wiki_articles else class_texts
        prototype     = np.mean(embedder.encode(source_texts), axis=0)
        label_prototypes.append(prototype)

    label_prototypes = np.array(label_prototypes)
    np.save(cache_path, label_prototypes)
    print(f"Saved WikiDoMiner prototype cache: {cache_path}")
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
