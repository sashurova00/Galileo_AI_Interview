"""
Topic modeling and clustering for support message intents.
"""

# intent_topics.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from bertopic import BERTopic
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_l1_topics(texts: List[str], embeddings: np.ndarray, min_topic_size: int = 12) -> Tuple[BERTopic, List[int], np.ndarray, pd.DataFrame]:
    """Cluster texts (L1) using BERTopic with precomputed embeddings."""
    topic_model = BERTopic(min_topic_size=min_topic_size, verbose=False, calculate_probabilities=True)
    topics, probs = topic_model.fit_transform(texts, embeddings=embeddings)
    info = topic_model.get_topic_info()  # columns: Topic, Count, Name
    return topic_model, topics, probs, info

def summarize_clusters(topic_model: BERTopic, topics: List[int], df: pd.DataFrame, max_words: int = 8, n_examples: int = 3) -> List[Dict[str, Any]]:
    out = []
    topic_ids = [t for t in topic_model.get_topic_info()["Topic"].tolist() if t != -1]
    for tid in topic_ids:
        words = [w for w,_ in topic_model.get_topic(tid)[:max_words]]
        idxs = np.where(np.array(topics) == tid)[0]
        exs = df.iloc[idxs]["text"].head(n_examples).tolist()
        out.append({
            "level": "L1",
            "topic_id": int(tid),
            "keywords": words,
            "size": int(len(idxs)),
            "example_texts": exs
        })
    return out

def run_l2_subtopics(parent_id: int, topics: List[int], embeddings: np.ndarray, min_size: int = 6) -> Dict[str, Any]:
    """Subcluster within a single L1 cluster using KMeans (robust/simple)."""
    idxs = np.where(np.array(topics) == parent_id)[0]
    if len(idxs) < (min_size * 2):
        return {"parent_id": int(parent_id), "labels": None, "indices": idxs.tolist()}  # too small to split
    X = embeddings[idxs]
    # pick k by size (2–5), use silhouette to choose best k
    candidates = [k for k in range(2, 6) if k < len(X)]
    best_k, best_score = None, -1
    for k in candidates:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X)
        score = silhouette_score(X, labels) if len(set(labels)) > 1 else -1
        if score > best_score:
            best_k, best_score = k, score
    if best_k is None:
        return {"parent_id": int(parent_id), "labels": None, "indices": idxs.tolist()}
    km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
    sublabels = km.labels_
    return {"parent_id": int(parent_id), "labels": sublabels.tolist(), "indices": idxs.tolist()}
