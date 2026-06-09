# Technical Report — Intelligent Candidate Ranking System

## India Runs by Redrob AI | Track 1: Intelligent Candidate Discovery

---

## 1. Problem Statement

Rank top 100 candidates from a pool of 100,000 for a given Job Description.

**Hard Constraints:**
- Zero network access during ranking
- Must complete under 5 minutes on CPU
- Output: `submission.csv` with exactly 100 rows

**What the hackathon actually wants** (from problem statement):
> *"Deep Job Understanding, Contextual Relevance — seeing beyond keywords to understand semantic fit, Signal Integration — leveraging all data: profile attributes, career metadata, and crucial activity/behavioral signals."*

---

## 2. Evolution of the System

### Version 1 — Pure Feature Engineering (11 features)

The first version extracted 11 features from each candidate and trained a RankNet model on them:

```
Features:
  github_activity_score, avg_skill_assessment_score,
  recruiter_response_rate, interview_completion_rate,
  offer_acceptance_rate, profile_completeness_score,
  years_of_experience, jd_skill_match_score,
  education_tier_score, recency_score, consulting_penalty
```

**What was wrong:**
- `jd_skill_match_score` counted JD keywords in the candidate's skills list
- Skills list is exactly the "trap" the hackathon warned about
- A Marketing Manager who listed "NLP, RAG, Pinecone" ranked high
- Ignored career history completely

---

### Version 2 — 14 Features (Added Availability Signals)

Added `notice_score`, `relocation_bonus`, `open_to_work` to V1.

**What was wrong:**
- Still used `jd_skill_match_score` (keyword trap)
- Career history descriptions still ignored
- Weight sum maintained at 1.0 but signal quality not improved

---

### Version 3 — Current: 2-Stage BM25 + RankNet

**Core insight from re-reading the problem statement:**

> *"Contextual Relevance: Seeing BEYOND keywords to understand semantic fit."*

This required a fundamentally different architecture.

---

## 3. Why 2-Stage Architecture?

### Stage 1: BM25 Retrieval

**What it does:** Scores all 100k candidates by semantic alignment of their job titles with the JD text.

**Why BM25 over keyword counting:**
- BM25 uses IDF weighting — rare, JD-specific terms like "retrieval", "ranking", "embedding" get higher weight than common words
- A candidate with title "Recommendation Systems Engineer" scores high because these are exactly the terms the JD uses
- A "Marketing Manager" scores near zero — their title has no JD-relevant terms
- This is the "contextual relevance" the hackathon wants

**Why titles only (not descriptions):**

During development, we ran a data inspection:
```python
# First 500 candidates had only 29 unique description snippets
# Descriptions were randomly assigned, not matching actual roles
Unique desc snippets in first 500 cands: 29
Unique role titles in first 500 cands: 36
```

Example of mismatch:
```
Operations Manager @ Wipro
Description: "Customer support team lead at a SaaS product..."

Marketing Manager @ Dunder Mifflin  
Description: "Mechanical engineering design role at a hardware company..."
```

The dataset has ~30 synthetic description templates randomly shuffled across candidates. **Using descriptions for BM25 = noise.**

Using titles gave:
- Build time: 2s (vs 149s with descriptions)
- Vocab size: 106 meaningful words (vs 1114 with stop-word noise)
- Top results: "Staff ML Engineer", "Senior NLP Engineer", "Recommendation Systems Engineer"

**Why NOT skills list:**
The hackathon explicitly called this out as a trap. We excluded it entirely.

---

### Stage 2: RankNet on Behavioral Features

**What it does:** Within the BM25-retrieved pool of 1000 candidates, re-ranks them using signals that reflect actual availability, quality, and career depth.

**Why RankNet (not simple ranking):**
- Pairwise learning-to-rank — learns relative ordering, not absolute scores
- Trained on synthetic pairs generated from a composite score
- Composite = BM25_score × 0.60 + behavioral_composite × 0.40
- This means the model learns "if candidate A has a higher blended signal than candidate B, A should rank higher"

---

## 4. Feature Design Decisions

### `yoe_sweet_spot` — Non-linear Experience Scoring

**Old approach:** Linear weight on years_of_experience. A 16-year veteran ranked #1.

**JD says:** *"Some people hit senior judgment at 4 years; some never hit it after 15. The ideal is 6-8 years."*

**New approach:** Bell curve peaking at 7 years:
```python
if yoe < 4:    return yoe / 4.0 * 0.55      # ramp up
if yoe <= 9:   return 1.0 - abs(yoe-7) / 7  # peak at 7
if yoe <= 14:  return 0.75 - (yoe-9)*0.03   # slight decay
else:          return 0.60                    # still ok
```

Result: Rank 1 moved from "Applied Scientist with 16.2 yrs" to "Staff ML Engineer with 5.7 yrs" — exactly the JD's sweet spot.

---

### `career_depth_score` — Production Evidence

**What it does:** Scans career descriptions for evidence of actual production ML work.

```python
PRODUCTION_EVIDENCE = {
    "production", "deployed", "shipped", "launched", "real users",
    "million", "billion", "ranking", "retrieval", "recommendation",
    "embedding", "vector", "a/b test", "ndcg", "pipeline", ...
}
```

**Why:** JD says *"A candidate who built a recommendation system for real users at a product company is a fit — even if they don't use the words RAG or Pinecone."*

Note: Because dataset descriptions are synthetic templates, this feature has limited discriminating power in this dataset — but it's the architecturally correct signal for real-world use.

---

### `consulting_penalty` — Per JD Explicit Instruction

**JD says:** *"People who have only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) in their entire career — we've had bad fit experiences."*

```python
CONSULTING_FIRMS = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture",
    "cognizant", "capgemini", "hcl", "tech mahindra"
}
# Returns fraction of career months at consulting firms (0.0 to 1.0)
# Applied as NEGATIVE weight in composite scoring
```

---

### `title_relevance_score` — Context Validation

**Problem it solves:** The "Marketing Manager trap" — someone with ML keywords in skills but wrong job context.

```python
ML_TITLE_SIGNALS = {
    "machine learning", "ml engineer", "nlp", "data scientist",
    "ai engineer", "recommendation", "search engineer", "retrieval",
    "applied scientist", ...
}
```

Score is based on current title + recent 3 career titles. A "Marketing Manager" gets 0 regardless of what skills they listed.

---

### `notice_score` — Availability Signal

**JD says:** *"We'd love sub-30-day notice. We can buy out up to 30 days. 30+ day notice candidates are still in scope but the bar gets higher."*

```python
notice_score = max(0.0, 1.0 - notice_days / 180.0)
# 0 days → 1.0 (ideal)
# 30 days → 0.83
# 90 days → 0.50
# 180 days → 0.0
```

---

## 5. What We Explicitly Removed vs V1/V2

| Feature | V1/V2 | V3 | Reason |
|---------|-------|-----|--------|
| `jd_skill_match_score` | ✅ Weight 0.16 | ❌ Removed | Keyword trap per hackathon |
| `avg_skill_assessment_score` | ✅ Weight 0.14 | ❌ Removed | Platform metric, seniors skip |
| `profile_completeness_score` | ✅ Weight 0.08 | ❌ Removed | Not correlated to ability |
| `offer_acceptance_rate` | ✅ Weight 0.05 | ❌ Removed | Not mentioned in JD |
| `education_tier_score` | ✅ Weight 0.07 | ❌ Removed | JD doesn't mention education |
| BM25 semantic retrieval | ❌ None | ✅ Stage 1 | Core "contextual relevance" |
| `yoe_sweet_spot` | ❌ Linear | ✅ Bell curve | JD has preferred range, not max |
| `title_relevance_score` | ❌ None | ✅ Added | Context validation |
| `career_depth_score` | ❌ None | ✅ Added | Production evidence |

---

## 6. Composite Weight Design

```python
COMPOSITE_WEIGHTS = np.array([
  # Feature                Weight   Rationale
  # github_activity_score  0.18    JD strongly values open-source
  # recruiter_response_rate 0.12   Availability
  # recency_score          0.09    Active on platform
  # interview_completion   0.08    Seriousness
  # notice_score           0.08    Can join soon
  # open_to_work           0.07    Actively seeking
  # relocation_bonus       0.06    Office location match
  # yoe_sweet_spot         0.11    JD experience preference
  # title_relevance        0.16    Role context validation
  # career_depth_score     0.20    Strongest non-keyword signal
  # consulting_penalty    -0.15    JD explicit disqualifier
  0.18, 0.12, 0.09, 0.08, 0.08, 0.07, 0.06, 0.11, 0.16, 0.20, -0.15
])
# Sum: 1.15 (positives) - 0.15 (penalty) = 1.00 ✓
```

---

## 7. Training Strategy

```
1. Load all 100k candidates
2. Build BM25 index on career titles
3. Score all 100k with BM25 → normalize to [0,1]
4. Extract 11 behavioral features → percentile normalize
5. Composite = BM25_norm × 0.60 + behavioral_composite × 0.40
6. Sample 50,000 pairs where |composite_A - composite_B| >= 0.08
7. Train RankNet on behavioral features:
   - Input: 11 features
   - Architecture: Linear(11→64) → ReLU → Dropout(0.1) → Linear(64→32) → ReLU → Linear(32→1) → Sigmoid
   - Loss: Pairwise Binary Cross-Entropy (RankNet loss)
   - Optimizer: Adam with CosineAnnealingLR
   - Epochs: 20, Batch: 512
8. Final loss: 0.3993 (lower = better pairwise ordering)
```

---

## 8. Inference Strategy

```
1. Build BM25 index (2s)
2. Score all 100k → get top-1000 by BM25
3. Extract behavioral features for ALL 100k
   → percentile normalize over full population (ensures same distribution as training)
4. Run RankNet on top-1000 behavioral features
5. Final score = BM25_norm × 0.60 + RankNet_score × 0.40
6. Sort top-1000, take top-100
7. Rescale scores to [0.20, 0.99]
8. Validate → write CSV
```

---

## 9. Results Comparison

| Metric | V1/V2 System | V3 (Current) |
|--------|-------------|--------------|
| Rank 1 | Applied Scientist, 16.2 yrs (over-band) | **Staff ML Engineer, 5.7 yrs** (JD sweet spot) |
| Rank 1 score | 0.9920 | 0.9900 |
| Total runtime | ~90s | **~25s** |
| Training time | ~497s | **~117s** |
| Final loss | 0.4135 | **0.3993** |
| Architecture | 1-stage keyword | **2-stage semantic** |
| Keyword matching | ✅ (trap!) | ❌ Excluded |
| Semantic retrieval | ❌ | ✅ BM25 |
| Validator | ✅ Passed | ✅ Passed |

---

## 10. Limitations & Future Improvements

### Current Limitations
1. **Synthetic descriptions:** Dataset descriptions don't match titles. `career_depth_score` has limited signal in this dataset but is architecturally correct for real data.
2. **Small vocab (106 words):** BM25 vocab is small due to title-only input. In production, richer text would give better discrimination.
3. **No cross-validation:** RankNet trained on full data without holdout set (acceptable for competition).

### What Would Improve It (Real World)
1. **Sentence Transformers:** Replace BM25 with dense semantic embeddings (pre-downloaded, offline). Much richer semantic understanding.
2. **Real career descriptions:** In production datasets, descriptions are genuine and would massively boost `career_depth_score`.
3. **LLM-based re-ranking:** Small offline LLM (e.g., Mistral-7B quantized) to score top-100 candidates against JD with chain-of-thought reasoning.
4. **Online feedback loop:** Recruiter click/engage data to continuously improve rankings.

---

## 11. File-by-File Summary

| File | Lines | Purpose |
|------|-------|---------|
| `src/bm25_retriever.py` | ~130 | BM25 implementation using scipy sparse matrices. Builds COO→CSR for fast construction. Tokenizes career titles. |
| `src/feature_extractor.py` | ~200 | Extracts 11 behavioral/career features. Percentile normalizes across full 100k population. No keyword matching. |
| `src/ranknet.py` | ~45 | RankNet architecture: 11→64→32→1. Pairwise BCE loss. Load/save utilities. |
| `src/train.py` | ~145 | Full training pipeline: load → BM25 → features → pairs → train → save model.pt |
| `src/rank.py` | ~240 | Inference: load → BM25 → top-1000 → features → RankNet → blend → validate → CSV |
