"""
feature_extractor.py  —  Stage 2: Behavioral + Career Signal Features
======================================================================
11 features — ZERO keyword matching.
JD skill keywords = trap. Yahan sirf actual quality/availability signals hain.

Features:
  0  github_activity_score     (code quality proxy)
  1  recruiter_response_rate   (availability signal)
  2  recency_score             (active on platform?)
  3  interview_completion_rate (seriousness signal)
  4  notice_score              (how soon can they join?)
  5  open_to_work              (actively seeking?)
  6  relocation_bonus          (willing to relocate to Pune/Noida?)
  7  yoe_sweet_spot            (JD says 5-9 yrs; bell curve around 7)
  8  title_relevance_score     (current/recent title ML/AI/Engineering?)
  9  career_depth_score        (deployed? shipped? real users? at scale?)
 10  consulting_penalty        (pure consulting career? NEGATIVE signal)
"""
from __future__ import annotations

import re
from datetime import date
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra",
}

# Titles that signal the candidate is actually in ML/AI/Engineering
# Not checking against JD keywords — checking if they DO this work
ML_TITLE_SIGNALS = {
    "machine learning", "ml engineer", "nlp", "data scientist",
    "ai engineer", "ai specialist", "applied scientist", "research engineer",
    "recommendation", "search engineer", "retrieval engineer",
    "deep learning", "applied ml", "computer vision", "speech",
    "staff ml", "senior ml", "principal ml", "junior ml",
    "software engineer", "backend engineer", "platform engineer",
    "infrastructure", "data engineer", "analytics engineer",
}

# What the JD ACTUALLY cares about — evidence of real production ML work
# Found in career_history[].description, NOT skills list
PRODUCTION_EVIDENCE = {
    # Shipped something real
    "production", "deployed", "shipped", "launched", "live",
    "real users", "at scale",
    # Numbers = real scale
    "million", "billion", "thousand", "users", "requests",
    # Core IR/ranking/search work
    "ranking", "retrieval", "recommendation", "search", "reranking",
    "embedding", "embeddings", "vector", "semantic", "similarity",
    "faiss", "elasticsearch", "opensearch", "pinecone",
    # Evaluation (senior engineers build this)
    "a/b test", "ndcg", "mrr", "offline evaluation", "online evaluation",
    # MLOps / pipeline
    "pipeline", "inference", "serving", "latency", "throughput",
    # Platform context
    "recruiter", "marketplace", "matching", "candidate",
}

_REFERENCE_ORD: int = date.today().toordinal()

NUM_FEATURES = 11
FEATURE_NAMES = [
    "github_activity_score",
    "recruiter_response_rate",
    "recency_score",
    "interview_completion_rate",
    "notice_score",
    "open_to_work",
    "relocation_bonus",
    "yoe_sweet_spot",
    "title_relevance_score",
    "career_depth_score",
    "consulting_penalty",
]

# Composite weights — must sum to 1.0
# consulting_penalty is NEGATIVE (higher penalty = worse candidate)
# Sum check: 0.16+0.11+0.08+0.07+0.07+0.06+0.05+0.10+0.14+0.16+(-0.15) = 0.85... hmm
# Let me recalculate:
# Positive: 0.16+0.11+0.08+0.07+0.07+0.06+0.05+0.10+0.14+0.16 = 1.00
# With penalty: 1.00 + (-0.15) = 0.85 — not 1.0
# Fix: scale positives to 1.15 so 1.15 - 0.15 = 1.00
# Positive budget = 1.15, distributed as:
COMPOSITE_WEIGHTS = np.array(
    # 0:github  1:rrr   2:recency 3:icr   4:notice 5:otw  6:reloc
    [  0.18,    0.12,   0.09,     0.08,   0.08,    0.07,  0.06,
    # 7:yoe_sw  8:title  9:depth  10:consulting
       0.11,    0.16,    0.20,   -0.15],
    dtype=np.float32,
)
# Verify: 0.18+0.12+0.09+0.08+0.08+0.07+0.06+0.11+0.16+0.20 = 1.15; -0.15 => 1.00


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _yoe_sweet_spot(yoe: float) -> float:
    """
    JD says 5-9 years preferred, ideal 6-8.
    Bell curve peaking at 7 years. Not a hard cutoff.
    """
    if yoe <= 0:
        return 0.0
    if yoe < 4:
        return yoe / 4.0 * 0.55          # ramp up: 0→0.55
    if yoe <= 9:
        # Peak 1.0 at 7 years
        return 1.0 - abs(yoe - 7.0) / 7.0
    if yoe <= 14:
        return 0.75 - (yoe - 9) * 0.03   # slight decay for over-experienced
    return 0.60                            # very experienced → still ok


def _title_relevance(candidate: dict[str, Any]) -> float:
    """
    Does the candidate's current title and recent roles signal actual
    ML/AI/Engineering work? Caps at 1.0.
    """
    score = 0.0
    profile = candidate.get("profile", {}) or {}

    current = (profile.get("current_title", "") or "").lower()
    for kw in ML_TITLE_SIGNALS:
        if kw in current:
            score += 0.6
            break

    # Check top-3 career roles
    for role in (candidate.get("career_history", []) or [])[:3]:
        role_title = (role.get("title", "") or "").lower()
        for kw in ML_TITLE_SIGNALS:
            if kw in role_title:
                score += 0.2
                break

    return min(score, 1.0)


def _career_depth(candidate: dict[str, Any]) -> float:
    """
    Scan career_history[].description for evidence of REAL production ML work.
    This is what the hackathon calls 'seeing beyond keywords' — someone who
    built a recommendation system for real users vs someone who listed
    'recommendation systems' as a skill.
    """
    full_text = " ".join(
        (role.get("description", "") or "")
        for role in (candidate.get("career_history", []) or [])
    ).lower()

    if not full_text.strip():
        return 0.0

    hits = sum(1 for kw in PRODUCTION_EVIDENCE if kw in full_text)
    return min(hits / 6.0, 1.0)           # 6+ hits → 1.0


def _consulting_penalty(candidate: dict[str, Any]) -> float:
    """Fraction of career at major IT consulting firms (0→1)."""
    history = candidate.get("career_history", []) or []
    if not history:
        return 0.0
    total = sum(r.get("duration_months", 0) for r in history) or 1
    consult = sum(
        r.get("duration_months", 0)
        for r in history
        if any(f in (r.get("company", "") or "").lower() for f in CONSULTING_FIRMS)
    )
    return min(consult / total, 1.0)


# ---------------------------------------------------------------------------
# Raw feature extraction
# ---------------------------------------------------------------------------

def extract_raw_features(candidate: dict[str, Any]) -> np.ndarray:
    """Extract 11 raw behavioral/career features. Shape (11,)."""
    signals = candidate.get("redrob_signals", {}) or {}
    profile  = candidate.get("profile", {}) or {}

    # 0 — GitHub activity (-1 → 0)
    github = float(signals.get("github_activity_score", 0) or 0)
    github = max(0.0, github)

    # 1 — Recruiter response rate
    rrr = float(signals.get("recruiter_response_rate", 0) or 0)
    rrr = max(0.0, min(1.0, rrr))

    # 2 — Recency (days since last active, inverted; cap 730 days)
    las = signals.get("last_active_date", "") or ""
    try:
        y, m, d = las.split("-")
        days_since = max(0, _REFERENCE_ORD - date(int(y), int(m), int(d)).toordinal())
    except Exception:
        days_since = 365
    recency = 1.0 - min(days_since, 730) / 730.0

    # 3 — Interview completion rate
    icr = float(signals.get("interview_completion_rate", 0) or 0)
    icr = max(0.0, min(1.0, icr))

    # 4 — Notice score (0 days→1.0, 180 days→0.0)
    nd = float(signals.get("notice_period_days", 90) or 90)
    notice = max(0.0, 1.0 - min(nd, 180.0) / 180.0)

    # 5 — Open to work
    otw = 1.0 if signals.get("open_to_work_flag", False) else 0.0

    # 6 — Relocation bonus (Pune/Noida office)
    reloc = 1.0 if signals.get("willing_to_relocate", False) else 0.0

    # 7 — YoE sweet spot
    yoe = float(profile.get("years_of_experience", 0) or 0)
    yoe_sw = _yoe_sweet_spot(yoe)

    # 8 — Title relevance
    title_rel = _title_relevance(candidate)

    # 9 — Career depth (production evidence in descriptions)
    depth = _career_depth(candidate)

    # 10 — Consulting penalty
    penalty = _consulting_penalty(candidate)

    return np.array(
        [github, rrr, recency, icr, notice, otw, reloc,
         yoe_sw, title_rel, depth, penalty],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Percentile normalization
# ---------------------------------------------------------------------------

def percentile_normalize(raw: np.ndarray) -> np.ndarray:
    """Percentile-normalise each column. (N, 11) → (N, 11) in [0,1]."""
    from scipy.stats import rankdata
    N, F = raw.shape
    out = np.zeros_like(raw, dtype=np.float32)
    denom = max(N - 1, 1)
    for j in range(F):
        out[:, j] = (rankdata(raw[:, j], method="average") - 1.0) / denom
    return out


# ---------------------------------------------------------------------------
# Composite scorer (for training pair generation)
# ---------------------------------------------------------------------------

def compute_composite(normed: np.ndarray) -> np.ndarray:
    """
    Weighted sum of normed features → composite score per candidate.
    consulting_penalty (col 10) has negative weight → penalises consultants.
    Returns shape (N,), clamped to [0, 1].
    """
    raw_score = normed @ COMPOSITE_WEIGHTS
    lo, hi = raw_score.min(), raw_score.max()
    if hi > lo:
        return ((raw_score - lo) / (hi - lo)).astype(np.float32)
    return np.zeros(len(raw_score), dtype=np.float32)


# ---------------------------------------------------------------------------
# Batch extractor
# ---------------------------------------------------------------------------

def extract_all_features(
    candidates: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    raw    : (N, 11) float32
    normed : (N, 11) float32 percentile-normalised
    """
    raw = np.stack([extract_raw_features(c) for c in candidates], axis=0)
    return raw, percentile_normalize(raw)
