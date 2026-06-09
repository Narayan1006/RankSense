"""
train.py  —  2-Stage Training Pipeline
=======================================
Stage 1: Build BM25 index, score all 100k candidates semantically
Stage 2: Extract 11 behavioral features, percentile-normalise
Composite = BM25_normed * 0.60 + behavioral_composite * 0.40
Generate pairwise training data, train RankNet on behavioral features only.

Run from project root:
  python src/train.py --candidates <candidates.jsonl> --jd data/job_description.txt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from bm25_retriever import BM25Retriever, build_candidate_text
from feature_extractor import extract_all_features, compute_composite
from ranknet import RankNet, NUM_FEATURES, pairwise_loss


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_candidates(path: str) -> list[dict]:
    print(f"[train] Loading {path} ...", flush=True)
    t0 = time.time()
    cands = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                cands.append(json.loads(line))
            if (i + 1) % 10_000 == 0:
                print(f"  {i+1:,} loaded ...", flush=True)
    print(f"  Total: {len(cands):,} in {time.time()-t0:.1f}s")
    return cands


# ---------------------------------------------------------------------------
# Pair generation
# ---------------------------------------------------------------------------

def generate_pairs(
    composite: np.ndarray,
    n_pairs: int = 50_000,
    threshold: float = 0.08,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample random candidate pairs where composite score differs by >= threshold.
    Returns (idx_i, idx_j) where i is the better candidate.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    N = len(composite)
    idx_i, idx_j = [], []
    attempts = 0
    max_attempts = n_pairs * 20

    while len(idx_i) < n_pairs and attempts < max_attempts:
        a, b = rng.integers(0, N, size=2)
        if abs(composite[a] - composite[b]) >= threshold:
            if composite[a] > composite[b]:
                idx_i.append(a); idx_j.append(b)
            else:
                idx_i.append(b); idx_j.append(a)
        attempts += 1

    return np.array(idx_i), np.array(idx_j)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    normed_matrix: np.ndarray,
    idx_i: np.ndarray,
    idx_j: np.ndarray,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
) -> RankNet:
    model   = RankNet(NUM_FEATURES)
    opt     = optim.Adam(model.parameters(), lr=lr)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    labels  = torch.ones(len(idx_i), dtype=torch.float32)

    feat_i  = torch.tensor(normed_matrix[idx_i], dtype=torch.float32)
    feat_j  = torch.tensor(normed_matrix[idx_j], dtype=torch.float32)
    ds      = TensorDataset(feat_i, feat_j, labels)
    loader  = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    params  = sum(p.numel() for p in model.parameters())
    print(f"\n[train] RankNet  features={NUM_FEATURES}  params={params:,}  epochs={epochs}")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        total_loss = 0.0
        model.train()
        for xi, xj, lbl in loader:
            opt.zero_grad()
            loss = pairwise_loss(model(xi), model(xj), lbl)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(lbl)
        sched.step()
        avg = total_loss / max(len(idx_i), 1)
        print(
            f"  Epoch {epoch:>2}/{epochs}  loss={avg:.6f}  "
            f"lr={sched.get_last_lr()[0]:.6f}  {time.time()-t0:.1f}s"
        )

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--jd",         required=True)
    ap.add_argument("--out",        default="models/model.pt")
    ap.add_argument("--pairs",      type=int,   default=50_000)
    ap.add_argument("--threshold",  type=float, default=0.08)
    ap.add_argument("--epochs",     type=int,   default=20)
    ap.add_argument("--top-k",      type=int,   default=2000,
                    help="BM25 top-K pool for pair generation (default 2000)")
    args = ap.parse_args()

    total_t0 = time.time()

    # 1. Load
    candidates = load_candidates(args.candidates)
    jd_text    = Path(args.jd).read_text(encoding="utf-8")
    N          = len(candidates)

    # 2. BM25 — Stage 1 semantic scoring
    print("\n[train] Building BM25 index ...", flush=True)
    t0     = time.time()
    corpus = [build_candidate_text(c) for c in candidates]
    bm25   = BM25Retriever().fit(corpus)
    print(f"[train] BM25 index built in {time.time()-t0:.1f}s", flush=True)

    print("[train] Scoring all candidates with BM25 ...", flush=True)
    t0         = time.time()
    bm25_raw   = bm25.score_all(jd_text)           # (N,)
    bm25_min, bm25_max = bm25_raw.min(), bm25_raw.max()
    bm25_norm  = (bm25_raw - bm25_min) / (bm25_max - bm25_min + 1e-9)
    print(f"[train] BM25 done in {time.time()-t0:.1f}s  "
          f"min={bm25_min:.2f}  max={bm25_max:.2f}", flush=True)

    # 3. Behavioral features — Stage 2
    print("\n[train] Extracting behavioral features ...", flush=True)
    t0             = time.time()
    raw, normed    = extract_all_features(candidates)
    behav_composite = compute_composite(normed)       # (N,) in [0,1]
    print(f"[train] Behavioral features done in {time.time()-t0:.1f}s")

    # 4. Composite = BM25 * 0.60 + behavioral * 0.40
    composite = 0.60 * bm25_norm + 0.40 * behav_composite
    print(f"[train] Composite  min={composite.min():.4f}  "
          f"max={composite.max():.4f}  mean={composite.mean():.4f}")

    # 5. Generate pairs
    print(f"\n[train] Generating {args.pairs:,} pairs ...", flush=True)
    t0           = time.time()
    idx_i, idx_j = generate_pairs(composite, args.pairs, args.threshold)
    print(f"[train] {len(idx_i):,} pairs in {time.time()-t0:.1f}s")

    # 6. Train RankNet on behavioral features
    model = train(normed, idx_i, idx_j, epochs=args.epochs)

    # 7. Save
    torch.save(model.state_dict(), args.out)
    print(f"\n[train] Model saved -> {args.out}")
    print(f"[train] Total time: {time.time()-total_t0:.1f}s")


if __name__ == "__main__":
    main()
