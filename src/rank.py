"""
rank.py  —  2-Stage Inference Pipeline
=======================================
Stage 1: BM25 scores all 100k candidates (semantic fit from career titles)
Stage 2: RankNet re-ranks top-1000 using behavioral features
Final:   blended score -> top 100 -> submission.csv

Run from project root:
  python src/rank.py --candidates <candidates.jsonl> \
                     --jd data/job_description.txt \
                     --out output/submission.csv \
                     --model models/model.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from bm25_retriever import BM25Retriever, build_candidate_text
from feature_extractor import extract_raw_features, percentile_normalize
from ranknet import load_model, predict_scores


# ---------------------------------------------------------------------------
# Reasoning string builder
# ---------------------------------------------------------------------------

def build_reasoning(candidate: dict, bm25_score: float, behav: np.ndarray) -> str:
    profile  = candidate.get("profile", {}) or {}
    signals  = candidate.get("redrob_signals", {}) or {}

    title    = profile.get("current_title", "Engineer") or "Engineer"
    yoe      = profile.get("years_of_experience", 0) or 0
    rrr      = signals.get("recruiter_response_rate", 0) or 0
    github   = signals.get("github_activity_score", 0) or 0
    notice   = signals.get("notice_period_days", 90) or 90
    otw      = signals.get("open_to_work_flag", False)

    # BM25 semantic percentile
    sem_pct  = int(round(bm25_score * 100))

    seeking  = "actively seeking" if otw else "passive"

    return (
        f"{title} with {yoe:.1f} yrs exp; "
        f"semantic fit top {100 - sem_pct}%; "
        f"github={int(github)}; "
        f"response rate {rrr:.2f}; "
        f"notice {int(notice)}d; "
        f"{seeking}"
    )


# ---------------------------------------------------------------------------
# Submission validator
# ---------------------------------------------------------------------------

def validate_submission(rows: list[dict]) -> list[str]:
    errors = []
    if len(rows) != 100:
        errors.append(f"Expected 100 rows, got {len(rows)}")
        return errors

    ranks  = [r["rank"] for r in rows]
    scores = [r["score"] for r in rows]

    if sorted(ranks) != list(range(1, 101)):
        errors.append("Ranks must be exactly 1-100")

    for i in range(len(rows) - 1):
        if scores[i] < scores[i + 1]:
            errors.append(
                f"Score not non-increasing at ranks "
                f"{rows[i]['rank']} ({scores[i]}) → {rows[i+1]['rank']} ({scores[i+1]})"
            )
            break

    # Tie-break: equal scores → ascending candidate_id
    for i in range(len(rows) - 1):
        if scores[i] == scores[i + 1]:
            if rows[i]["candidate_id"] > rows[i + 1]["candidate_id"]:
                errors.append(
                    f"Equal scores at ranks {rows[i]['rank']} and "
                    f"{rows[i+1]['rank']}: tie-break requires candidate_id ascending "
                    f"('{rows[i]['candidate_id']}' > '{rows[i+1]['candidate_id']}')"
                )
                break
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--jd",         required=True)
    ap.add_argument("--out",        default="output/submission.csv")
    ap.add_argument("--model",      default="models/model.pt")
    ap.add_argument("--top-k",      type=int, default=1000,
                    help="BM25 pool size before RankNet re-ranking (default 1000)")
    ap.add_argument("--bm25-weight", type=float, default=0.60,
                    help="Weight for BM25 score in final blend (default 0.60)")
    args = ap.parse_args()

    total_t0 = time.time()

    # 1. Load candidates
    print(f"[rank] Loading {args.candidates} ...", flush=True)
    t0         = time.time()
    candidates = []
    with open(args.candidates, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
            if (i + 1) % 20_000 == 0:
                print(f"  {i+1:,} loaded ...", flush=True)
    N = len(candidates)
    print(f"  Total: {N:,} in {time.time()-t0:.1f}s")

    # 2. Read JD
    jd_text = Path(args.jd).read_text(encoding="utf-8")
    print(f"[rank] JD length: {len(jd_text)} chars")

    # 3. Stage 1 — BM25
    print("\n[rank] Stage 1: Building BM25 index ...", flush=True)
    t0     = time.time()
    corpus = [build_candidate_text(c) for c in candidates]
    bm25   = BM25Retriever().fit(corpus)
    print(f"[rank] BM25 index built in {time.time()-t0:.1f}s")

    print(f"[rank] Scoring {N:,} candidates with BM25 ...", flush=True)
    t0        = time.time()
    bm25_raw  = bm25.score_all(jd_text)
    b_min, b_max = bm25_raw.min(), bm25_raw.max()
    bm25_norm = (bm25_raw - b_min) / (b_max - b_min + 1e-9)
    print(f"[rank] BM25 done in {time.time()-t0:.1f}s")
    print(f"[rank] BM25 stats: min={b_min:.2f}  max={b_max:.2f}")

    # Get top-K indices by BM25
    K          = min(args.top_k, N)
    top_k_idx  = np.argpartition(bm25_raw, -K)[-K:]
    top_k_idx  = top_k_idx[np.argsort(-bm25_raw[top_k_idx])]
    print(f"[rank] BM25 top-{K} selected")

    # 4. Stage 2 — Behavioral feature extraction (full population for percentile)
    print("\n[rank] Stage 2: Extracting behavioral features ...", flush=True)
    t0  = time.time()
    raw = np.stack([extract_raw_features(c) for c in candidates], axis=0)
    normed = percentile_normalize(raw)          # percentile over full 100k
    print(f"[rank] Behavioral features done in {time.time()-t0:.1f}s")

    # 5. RankNet on top-K behavioral features
    print(f"[rank] Loading model {args.model} ...", flush=True)
    model        = load_model(args.model, device="cpu")
    t0           = time.time()
    topk_normed  = normed[top_k_idx]            # (K, 11)
    ranknet_sc   = predict_scores(model, topk_normed)   # (K,)
    print(f"[rank] RankNet scoring done in {time.time()-t0:.1f}s")

    # 6. Blend: BM25 (semantic) + RankNet (behavioral)
    bm25_w  = args.bm25_weight
    rank_w  = 1.0 - bm25_w
    blended = bm25_w * bm25_norm[top_k_idx] + rank_w * ranknet_sc

    print(f"[rank] Blended stats: min={blended.min():.4f}  max={blended.max():.4f}")

    # 7. Sort by blended, tie-break by candidate_id
    cand_ids = [candidates[i]["candidate_id"] for i in top_k_idx]
    order    = sorted(
        range(K),
        key=lambda k: (-blended[k], cand_ids[k])
    )
    top100_pos = order[:100]

    # 8. Build rows with rescaled scores [0.20, 0.99]
    top100_scores = [float(blended[p]) for p in top100_pos]
    lo, hi = min(top100_scores), max(top100_scores)
    span   = hi - lo if hi > lo else 1.0

    def rescale(v: float) -> float:
        return round(0.20 + 0.79 * (v - lo) / span, 4)

    rows = []
    for pos in top100_pos:
        cand_idx = top_k_idx[pos]
        rows.append({
            "candidate_id": candidates[cand_idx]["candidate_id"],
            "rank":         0,                  # assigned after sort
            "score":        rescale(float(blended[pos])),
            "reasoning":    build_reasoning(
                candidates[cand_idx],
                bm25_norm[cand_idx],
                normed[cand_idx],
            ),
        })

    # 9. Final sort (score desc, tie-break candidate_id asc) + assign ranks
    rows.sort(key=lambda r: (-r["score"], r["candidate_id"]))
    for i, row in enumerate(rows):
        row["rank"] = i + 1

    # 10. Validate
    errors = validate_submission(rows)
    if errors:
        print("\n[rank] ** Validation FAILED **")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    print("[rank] Validation passed")

    # 11. Write CSV
    out_path = Path(args.out)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[rank] Written -> {out_path}")
    print("[rank] Top 5:")
    for row in rows[:5]:
        print(f"  Rank {row['rank']:>3}: {row['candidate_id']}  "
              f"score={row['score']:.4f}  {row['reasoning'][:70]}")

    elapsed = time.time() - total_t0
    print(f"\n[rank] Total runtime: {elapsed:.1f}s")
    if elapsed > 300:
        print("[rank] WARNING: Exceeded 5-minute limit!")


if __name__ == "__main__":
    main()
