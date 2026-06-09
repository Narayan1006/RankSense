"""
bm25_retriever.py  —  Stage 1: Semantic Retrieval
==================================================
IMPORTANT INSIGHT: Dataset descriptions are synthetic templates (only ~30 unique
descriptions for 100k candidates, randomly assigned). Using descriptions for BM25
would add noise, not signal.

RELIABLE signals in this dataset:
  - career_history[].title  (the actual role the candidate held)
  - profile.current_title
  - skills[].name           (but this is the keyword trap — use carefully)

BM25 here scores candidates by how well their JOB TITLES align with JD language.
A candidate who has held "NLP Engineer", "Search Engineer", "Recommendation Systems"
roles will score higher than one who held "Marketing Manager", "Accountant" roles.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix

# ---------------------------------------------------------------------------
_STOP = frozenset({
    "the","and","for","with","in","of","to","an","is","was","are",
    "were","be","been","have","has","had","do","does","did",
    "will","would","could","should","may","might","can","on","at","by",
    "from","as","or","but","not","so","if","this","that","it","its",
    "we","you","he","she","they","their","also","all","into","up",
    "out","about","how","what","which","who","when","where","very",
    "over","other","our","my","such","than","just","some","then",
    "there","these","those","them","own","same",
})

_RE = re.compile(r"[a-z][a-z0-9]{1,}")


def _tokenize(text: str) -> list[str]:
    return [t for t in _RE.findall(text.lower())
            if t not in _STOP and len(t) > 1]


# ---------------------------------------------------------------------------

def build_candidate_text(candidate: dict[str, Any]) -> str:
    """
    Build candidate document from TITLES only.

    WHY NOT DESCRIPTIONS: Dataset has only ~30 unique synthetic description
    templates randomly assigned to candidates. Descriptions don't match titles.
    Using descriptions adds noise. Titles are the reliable signal.

    WHY NOT SKILLS LIST: Explicitly the "keyword trap" per hackathon problem statement.

    WHAT WE USE:
      - current_title (4x weight — most important signal)
      - profile.headline
      - career_history[].title (recent roles weighted more)
      - certifications[].name (light signal)
    """
    parts: list[str] = []
    profile = candidate.get("profile", {}) or {}

    # Current title — strongest signal, 4x weight
    ct = profile.get("current_title", "") or ""
    if ct:
        parts.extend([ct] * 4)

    # Headline
    hl = profile.get("headline", "") or ""
    if hl:
        parts.append(hl)

    # Career titles — recent roles get more weight
    for i, role in enumerate((candidate.get("career_history", []) or [])):
        w     = max(1, 3 - i)           # role 0=×3, role 1=×2, rest=×1
        title = role.get("title", "") or ""
        if title:
            parts.extend([title] * w)

    # Certifications (light)
    for cert in (candidate.get("certifications", []) or []):
        n = cert.get("name", "") or ""
        if n:
            parts.append(n)

    return " ".join(parts)


# ---------------------------------------------------------------------------

class BM25Retriever:
    """
    Okapi BM25 — fast COO->CSR sparse matrix build.
    Scores candidates by title-level semantic alignment with JD.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b  = b
        self.vocab: dict[str, int] = {}
        self.idf: np.ndarray | None = None
        self.tf_matrix = None
        self.doc_lengths: np.ndarray | None = None
        self.avgdl: float = 1.0

    def fit(self, corpus: list[str]) -> "BM25Retriever":
        import time
        N         = len(corpus)
        t0        = time.time()
        tokenized = [_tokenize(doc) for doc in corpus]

        vocab_set: set[str] = set()
        for toks in tokenized:
            vocab_set.update(toks)
        self.vocab = {w: i for i, w in enumerate(sorted(vocab_set))}
        V          = len(self.vocab)

        self.doc_lengths = np.array([len(t) for t in tokenized], dtype=np.float32)
        self.avgdl       = float(self.doc_lengths.mean()) if N else 1.0

        # COO → CSR (fast for 100k docs)
        row_l, col_l, val_l = [], [], []
        df = np.zeros(V, dtype=np.float32)

        for i, toks in enumerate(tokenized):
            cnt = Counter(self.vocab[t] for t in toks if t in self.vocab)
            for ci, freq in cnt.items():
                row_l.append(i); col_l.append(ci); val_l.append(float(freq))
                df[ci] += 1.0

        self.tf_matrix = csr_matrix(
            (np.array(val_l, dtype=np.float32),
             (np.array(row_l, dtype=np.int32),
              np.array(col_l, dtype=np.int32))),
            shape=(N, V), dtype=np.float32,
        )
        self.idf = np.log((N - df + 0.5) / (df + 0.5) + 1.0).astype(np.float32)
        print(f"  [BM25] vocab={V:,}  docs={N:,}  built in {time.time()-t0:.1f}s",
              flush=True)
        return self

    def score_all(self, query: str) -> np.ndarray:
        qidxs = list({self.vocab[t] for t in _tokenize(query) if t in self.vocab})
        if not qidxs:
            return np.zeros(self.tf_matrix.shape[0], dtype=np.float32)
        tf_sub  = self.tf_matrix[:, qidxs].toarray().astype(np.float32)
        idf_sub = self.idf[qidxs]
        dl_norm = (1 - self.b + self.b * self.doc_lengths / self.avgdl)[:, None]
        bm25_tf = tf_sub * (self.k1 + 1.0) / (tf_sub + self.k1 * dl_norm)
        return (bm25_tf @ idf_sub).astype(np.float32)

    def top_k(self, query: str, k: int = 1000) -> tuple[np.ndarray, np.ndarray]:
        scores = self.score_all(query)
        idx    = np.argpartition(scores, -k)[-k:]
        idx    = idx[np.argsort(-scores[idx])]
        return idx, scores[idx]
