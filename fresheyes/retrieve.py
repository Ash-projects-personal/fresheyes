"""Passage retrieval over the agent's own source documents.

Pure-Python TF-IDF, no external model and no embeddings. The corpus is the
agent's own documents, not the web: that is the whole point of FreshEyes.
Retrieval supplies the evidence passages a verdict quotes; it never decides
a verdict by itself. The checkers in checkers.py decide, by matching numbers
and quotes against the retrieved passages.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .extract import STOPWORDS, content_words
from .models import SourceDoc


@dataclass
class Passage:
    doc_id: str
    title: str
    text: str
    score: float = 0.0


def _passages(doc: SourceDoc) -> list[Passage]:
    out = []
    for line in doc.text.splitlines():
        line = line.strip()
        if line:
            out.append(Passage(doc_id=doc.doc_id, title=doc.title, text=line))
    return out


class Retriever:
    """TF-IDF over source-document lines, built once per verification run."""

    def __init__(self, sources: list[SourceDoc]):
        self.passages: list[Passage] = []
        for doc in sources:
            self.passages.extend(_passages(doc))
        self._idf: dict[str, float] = {}
        self._vectors: list[dict[str, float]] = []
        self._build()

    def _build(self) -> None:
        n = len(self.passages)
        df: dict[str, int] = {}
        tf_list: list[dict[str, float]] = []
        for p in self.passages:
            words = content_words(p.text)
            tf: dict[str, float] = {}
            for w in words:
                tf[w] = tf.get(w, 0.0) + 1.0
            for w in tf:
                df[w] = df.get(w, 0) + 1
            tf_list.append(tf)
        for w, d in df.items():
            self._idf[w] = math.log((n + 1) / (d + 0.5)) + 1.0
        for tf in tf_list:
            vec = {w: c * self._idf[w] for w, c in tf.items()}
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            self._vectors.append({w: v / norm for w, v in vec.items()})

    def search(self, query: str, top_k: int = 3) -> list[Passage]:
        qwords = content_words(query)
        qvec: dict[str, float] = {}
        for w in qwords:
            if w in self._idf:
                qvec[w] = qvec.get(w, 0.0) + self._idf[w]
        norm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        qvec = {w: v / norm for w, v in qvec.items()}
        scored = []
        for p, vec in zip(self.passages, self._vectors):
            score = sum(qvec.get(w, 0.0) * v for w, v in vec.items())
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, p in scored[:top_k]:
            out.append(Passage(doc_id=p.doc_id, title=p.title, text=p.text, score=round(score, 3)))
        return out

    def search_doc(self, doc_id: str, query: str, top_k: int = 3) -> list[Passage]:
        """Retrieve within one cited document only."""
        sub = Retriever.__new__(Retriever)
        sub.passages = [p for p in self.passages if p.doc_id == doc_id]
        sub._idf = {}
        sub._vectors = []
        sub._build()
        return sub.search(query, top_k=top_k)


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace and punctuation, for quote matching."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
