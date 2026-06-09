---
title: RankSense AI
emoji: 🎯
colorFrom: red
colorTo: orange
sdk: streamlit
sdk_version: 1.41.1
app_file: app.py
pinned: true
short_description: Intelligent candidate ranking — BM25 + RankNet, zero keyword matching
---

# Intelligent Candidate Ranking System
### India Runs by Redrob AI — Track 1: Intelligent Candidate Discovery

---

## Overview

A **2-stage offline candidate ranking system** that ranks 100,000 candidates for a given Job Description in under **25 seconds on CPU** with zero network access.

Goes beyond keyword matching using **BM25 semantic retrieval on career titles** (Stage 1) + **behavioral RankNet re-ranking** (Stage 2).

> *"The right answer is NOT finding candidates whose skills section contains the most AI keywords. That's a trap."* — Hackathon Problem Statement

---

## Project Structure

```
Ranker_system/
├── src/
│   ├── bm25_retriever.py       ← Stage 1: BM25 semantic retrieval
│   ├── feature_extractor.py    ← Stage 2: 11 behavioral features
│   ├── ranknet.py              ← Neural LTR model
│   ├── train.py                ← Training pipeline
│   └── rank.py                 ← Inference → submission.csv
├── models/
│   └── model.pt                ← Pre-trained RankNet weights
├── data/
│   └── job_description.txt     ← JD for training/ranking
├── output/
│   └── submission.csv          ← Top 100 candidates
├── README.md
└── requirements.txt
```

---

## Architecture

```
100,000 Candidates
       │
       ▼
  STAGE 1: BM25 Retrieval        ← Career titles vs JD
  (filters irrelevant profiles)
       │ Top-1000
       ▼
  STAGE 2: RankNet Re-ranking    ← 11 behavioral signals
       │ Blended score
       ▼
  Top 100 → submission.csv
```

---

## Stage 2 Features (11, Zero Keywords)

| # | Feature | Signal |
|---|---------|--------|
| 0 | `github_activity_score` | Real code contributions |
| 1 | `recruiter_response_rate` | Responds to outreach? |
| 2 | `recency_score` | Active on platform? |
| 3 | `interview_completion_rate` | Serious about process? |
| 4 | `notice_score` | Can join soon? (JD: sub-30d preferred) |
| 5 | `open_to_work` | Actively seeking? |
| 6 | `relocation_bonus` | Willing to relocate to Pune/Noida? |
| 7 | `yoe_sweet_spot` | Bell curve at 7yrs (JD: 5-9 preferred) |
| 8 | `title_relevance_score` | Current title ML/AI/Engineering? |
| 9 | `career_depth_score` | Production ML evidence in descriptions |
| 10 | `consulting_penalty` | Pure consulting career? (JD penalizes) |

---

## Setup & Usage

```bash
pip install -r requirements.txt
```

**Run inference (from project root):**
```bash
python src/rank.py \
  --candidates <path/to/candidates.jsonl> \
  --jd data/job_description.txt \
  --out output/submission.csv \
  --model models/model.pt
```

**Re-train model:**
```bash
python src/train.py \
  --candidates <path/to/candidates.jsonl> \
  --jd data/job_description.txt \
  --out models/model.pt
```

| Flag | Default | Description |
|------|---------|-------------|
| `--top-k` | 1000 | BM25 pool size |
| `--bm25-weight` | 0.60 | BM25 vs RankNet blend |
| `--epochs` | 20 | Training epochs |

---

## Performance

| Step | Time |
|------|------|
| Load 100k candidates | ~8s |
| BM25 index build | ~2s |
| BM25 score all 100k | <0.1s |
| Behavioral features | ~18s |
| RankNet inference | <1s |
| **Total** | **~25s** ✅ |

---

## Output

```csv
candidate_id,rank,score,reasoning
CAND_0060072,1,0.9900,"Staff Machine Learning Engineer with 5.7 yrs exp; semantic fit top 0%; github=82; ..."
```

Validate: `python validate_submission.py output/submission.csv`

---

## Key Discovery

Dataset has **~30 synthetic description templates** randomly assigned — descriptions don't match actual job titles.

- ❌ BM25 on descriptions = noise (149s build, poor signal)
- ✅ BM25 on **titles** = reliable (2s build, strong signal)
- ❌ Skills list = keyword trap (excluded entirely)

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for full analysis.

---

## Top 5 Ranked Candidates

| Rank | Candidate ID | Title | Exp | Score | Semantic Fit | Key Signal |
|------|-------------|-------|-----|-------|-------------|------------|
| 1 | CAND_0060072 | Staff Machine Learning Engineer | 5.7 yrs | 0.9900 | Top 0% | GitHub 82, in JD sweet spot |
| 2 | CAND_0086022 | Senior Applied Scientist | 5.3 yrs | 0.8439 | Top 8% | GitHub 75, actively seeking |
| 3 | CAND_0037980 | Senior Applied Scientist | 9.0 yrs | 0.7159 | Top 14% | GitHub 30, actively seeking |
| 4 | CAND_0046064 | Senior NLP Engineer | 8.9 yrs | 0.6942 | Top 15% | GitHub 67, 30d notice |
| 5 | CAND_0077337 | Staff Machine Learning Engineer | 7.0 yrs | 0.6843 | Top 16% | GitHub 68, 95% response rate |

**All top candidates are:**
- In the JD's preferred **5–9 year experience band**
- Holding genuine **ML/NLP/Applied Science titles** — not keyword-stuffed profiles
- **Actively available** (open to work, reasonable notice periods)

> Compare with the old keyword-based system which ranked a 16.2-year Applied Scientist at #1 — technically over the JD's preferred band and not the ideal fit.
