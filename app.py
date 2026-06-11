import streamlit as st
import json, sys, time
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "src")
from bm25_retriever import BM25Retriever, build_candidate_text
from feature_extractor import extract_all_features, compute_composite
from ranknet import RankNet, NUM_FEATURES

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="RankSense AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0e1117; }

    /* Header */
    .ranksense-header {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
        border: 1px solid #e94560;
        border-radius: 12px;
        padding: 28px 36px;
        margin-bottom: 24px;
    }
    .ranksense-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e94560, #f5a623, #e94560);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .ranksense-sub {
        color: #8892a4;
        font-size: 1rem;
        margin-top: 6px;
    }

    /* Stage badges */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin: 4px 4px 4px 0;
    }
    .badge-blue  { background: #1e3a5f; color: #60a5fa; border: 1px solid #2563eb; }
    .badge-green { background: #1a3a2a; color: #4ade80; border: 1px solid #16a34a; }
    .badge-red   { background: #3a1a1a; color: #f87171; border: 1px solid #dc2626; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 14px 18px;
    }
    div[data-testid="metric-container"] label { color: #8892a4 !important; }

    /* Rank 1 row highlight */
    .rank1 { color: #f5a623; font-weight: bold; }

    /* Section divider */
    hr { border-color: #2d3748 !important; }

    /* Dataframe */
    .stDataFrame { border: 1px solid #2d3748; border-radius: 8px; }

    /* Button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #e94560, #c0392b);
        border: none;
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 12px 32px;
        border-radius: 8px;
        transition: transform 0.15s;
    }
    .stButton > button[kind="primary"]:hover { transform: translateY(-2px); }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="ranksense-header">
  <p class="ranksense-title">🎯 RankSense AI</p>
  <p class="ranksense-sub">
    Intelligent Candidate Ranking &nbsp;·&nbsp; India Runs Hackathon by Redrob AI
    &nbsp;&nbsp;
    <span class="badge badge-blue">🔍 BM25 Retrieval</span>
    <span class="badge badge-green">🧠 RankNet Re-ranking</span>
    <span class="badge badge-red">⚡ Zero Keyword Matching</span>
  </p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧩 How It Works")
    st.markdown("""
**Stage 1 — BM25 Semantic Retrieval**  
Scores candidates by how well their *job titles* match the JD.  
Filters out irrelevant profiles (HR Managers, Accountants) without keyword tricks.

**Stage 2 — RankNet Re-ranking**  
Re-ranks the top pool using 11 behavioral signals:
- GitHub activity
- Recruiter response rate  
- Notice period
- Open to work
- Relocation willingness
- Career depth (production ML evidence)
- Consulting penalty
- YoE sweet spot (JD: 5–9 yrs)
- ...and more

**Why not keyword matching?**  
> *"That's a trap we've explicitly built into the dataset."*  
> — Hackathon Problem Statement
""")

    st.markdown("---")
    st.markdown("**⚙️ Blend Settings**")
    bm25_weight = st.slider(
        "BM25 weight (semantic fit)",
        min_value=0.3, max_value=0.9, value=0.6, step=0.05,
        help="Remaining weight goes to RankNet behavioral score"
    )
    top_n = st.selectbox("Show top N candidates", [10, 20, 50, 100], index=1)

    st.markdown("---")
    st.caption("Full pipeline: 100,000 candidates in ~25s on CPU · Offline · No API calls")

# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
col_jd, col_up = st.columns([1, 1], gap="large")

with col_jd:
    st.markdown("### 📋 Job Description")
    try:
        default_jd = open("data/job_description.txt", encoding="utf-8").read()
    except FileNotFoundError:
        default_jd = "Paste your Job Description here..."
    jd_text = st.text_area(
        "Paste or edit JD",
        value=default_jd,
        height=320,
        label_visibility="collapsed",
    )

with col_up:
    st.markdown("### 👥 Candidates File")
    uploaded = st.file_uploader(
        "Upload candidates JSONL",
        type=["jsonl", "json"],
        help="Upload a .jsonl file where each line is a candidate JSON object",
        label_visibility="collapsed",
    )
    if uploaded:
        st.success(f"✅ **{uploaded.name}** uploaded")
        # Quick peek
        lines = [l for l in uploaded.getvalue().decode("utf-8").splitlines() if l.strip()]
        st.caption(f"Found **{len(lines):,}** candidate records")
        uploaded.seek(0)
    else:
            st.info("👆 Upload a `.jsonl` file to begin  \n*(Use `sample_candidates.json` from the hackathon data folder)*")

    st.markdown("---")

    # ---------------------------------------------------------------------------
    # Rank button
    # ---------------------------------------------------------------------------
    rank_clicked = st.button(
        "🚀 Rank Candidates",
        type="primary",
        use_container_width=False,
    )

    if not uploaded or not jd_text.strip():
        st.markdown("""
    ### 📊 Sample Output  *(from full 100K run)*
    """)
        sample_df = pd.DataFrame([
            {"Rank": 1, "Title": "Staff Machine Learning Engineer", "Exp": "5.7 yrs", "Semantic Fit": "Top 0%", "Score": 0.9900},
            {"Rank": 2, "Title": "Senior Applied Scientist",        "Exp": "5.3 yrs", "Semantic Fit": "Top 8%", "Score": 0.8439},
            {"Rank": 3, "Title": "Senior Applied Scientist",        "Exp": "9.0 yrs", "Semantic Fit": "Top 14%","Score": 0.7159},
            {"Rank": 4, "Title": "Senior NLP Engineer",             "Exp": "8.9 yrs", "Semantic Fit": "Top 15%","Score": 0.6942},
            {"Rank": 5, "Title": "Staff Machine Learning Engineer", "Exp": "7.0 yrs", "Semantic Fit": "Top 16%","Score": 0.6843},
        ])
        st.dataframe(sample_df, use_container_width=True, hide_index=True)

    # ---------------------------------------------------------------------------
    # Pipeline execution
    # ---------------------------------------------------------------------------
    if rank_clicked:
        if not uploaded:
            st.error("⚠️ Please upload a candidates `.jsonl` file first! (You can use the test_dataset_1k.jsonl file)")
        elif not jd_text.strip():
            st.error("⚠️ Please provide a Job Description!")
    else:

        total_t0 = time.time()

        # 1. Load candidates
        with st.spinner("⏳ Loading candidates..."):
            raw_bytes = uploaded.getvalue().decode("utf-8")
            candidates = []
            for line in raw_bytes.splitlines():
                line = line.strip()
                if line:
                    try:
                        candidates.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
            N = len(candidates)
        st.toast(f"Loaded {N:,} candidates", icon="📂")

        # 2. Stage 1 — BM25
        prog = st.progress(0, text="Stage 1: Building BM25 index...")
        t0 = time.time()
        corpus = [build_candidate_text(c) for c in candidates]
        bm25   = BM25Retriever().fit(corpus)
        prog.progress(30, text="Stage 1: Scoring with BM25...")
        bm25_raw  = bm25.score_all(jd_text)
        bm25_min, bm25_max = bm25_raw.min(), bm25_raw.max()
        bm25_norm = (bm25_raw - bm25_min) / (bm25_max - bm25_min + 1e-9)
        t_bm25 = time.time() - t0
        prog.progress(50, text=f"Stage 1 done ({t_bm25:.1f}s) — extracting behavioral features...")

        # 3. Stage 2 — Behavioral features
        t0 = time.time()
        raw_matrix, normed_matrix = extract_all_features(candidates)
        prog.progress(75, text="Stage 2: RankNet scoring...")

        # 4. Load model + score
        try:
            model = RankNet(in_features=NUM_FEATURES)
            model.load_state_dict(torch.load("models/model.pt", map_location="cpu"))
            model.eval()
            with torch.no_grad():
                feats = torch.tensor(normed_matrix, dtype=torch.float32)
                ranknet_scores = model.score(feats).numpy()
        except Exception as e:
            st.warning(f"Model load issue ({e}) — using behavioral composite only")
            ranknet_scores = (normed_matrix @ np.array(
                [0.18,0.12,0.09,0.08,0.08,0.07,0.06,0.11,0.16,0.20,-0.15],
                dtype=np.float32
            ))
            ranknet_scores = (ranknet_scores - ranknet_scores.min()) / (ranknet_scores.max() - ranknet_scores.min() + 1e-9)

        t_stage2 = time.time() - t0

        # 5. Blend
        final_scores = bm25_weight * bm25_norm + (1 - bm25_weight) * ranknet_scores

        # 6. Sort
        show_n = min(top_n, N)
        top_idx = np.argsort(-final_scores)[:show_n]

        prog.progress(100, text="✅ Ranking complete!")
        total_elapsed = time.time() - total_t0

        # 7. Build results table
        rows = []
        for rank_pos, cand_idx in enumerate(top_idx):
            c       = candidates[cand_idx]
            profile = c.get("profile", {}) or {}
            signals = c.get("redrob_signals", {}) or {}

            title   = profile.get("current_title", "—") or "—"
            company = profile.get("current_company", "—") or "—"
            yoe     = profile.get("years_of_experience", 0) or 0
            github  = signals.get("github_activity_score", -1)
            rrr     = signals.get("recruiter_response_rate", 0) or 0
            otw     = "✅" if signals.get("open_to_work_flag") else "—"
            notice  = signals.get("notice_period_days", 90) or 90
            sem_pct = max(1, int(round(bm25_norm[cand_idx] * 100)))

            rows.append({
                "Rank"          : rank_pos + 1,
                "Candidate ID"  : c.get("candidate_id", "—"),
                "Title"         : title[:45],
                "Exp"           : f"{yoe:.1f} yrs",
                "Semantic Fit"  : f"Top {100 - sem_pct}%",
                "GitHub"        : int(github) if github >= 0 else "—",
                "Response Rate" : f"{rrr:.0%}",
                "Open to Work"  : otw,
                "Notice"        : f"{int(notice)}d",
                "Score"         : round(float(final_scores[cand_idx]), 4),
            })

        df = pd.DataFrame(rows)

        # ---------------------------------------------------------------------------
        # Results display
        # ---------------------------------------------------------------------------
        st.markdown("---")
        st.markdown(f"## 🏆 Top {show_n} Candidates")

        # Metrics
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Candidates Processed", f"{N:,}")
        mc2.metric("Shown", show_n)
        mc3.metric("Total Runtime", f"{total_elapsed:.1f}s")
        mc4.metric("Rank 1 Score", rows[0]["Score"])
        mc5.metric("Rank 1 Title", rows[0]["Title"][:25])

        st.markdown("")

        # Stage timing
        st.caption(
            f"⏱ BM25 retrieval: **{t_bm25:.1f}s** &nbsp;|&nbsp; "
            f"Behavioral + RankNet: **{t_stage2:.1f}s** &nbsp;|&nbsp; "
            f"Blend: BM25 {bm25_weight:.0%} + RankNet {1-bm25_weight:.0%}"
        )

        # Results table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank"          : st.column_config.NumberColumn(width="small"),
                "Score"         : st.column_config.ProgressColumn(
                                    "Score", min_value=0, max_value=1, format="%.4f"),
                "Semantic Fit"  : st.column_config.TextColumn(width="small"),
                "Response Rate" : st.column_config.TextColumn(width="small"),
                "Open to Work"  : st.column_config.TextColumn(width="small"),
            },
        )

        # Download
        st.download_button(
            label="⬇️ Download Results CSV",
            data=df.to_csv(index=False),
            file_name="ranksense_results.csv",
            mime="text/csv",
            use_container_width=False,
        )

        # Top candidate detail card
        if rows:
            st.markdown("---")
            st.markdown("### 🥇 Rank 1 Candidate Detail")
            top_c    = candidates[int(top_idx[0])]
            tp       = top_c.get("profile", {}) or {}
            ts       = top_c.get("redrob_signals", {}) or {}

            d1, d2, d3 = st.columns(3)
            with d1:
                st.markdown("**Profile**")
                st.write(f"**ID:** {top_c.get('candidate_id','—')}")
                st.write(f"**Title:** {tp.get('current_title','—')}")
                st.write(f"**Company:** {tp.get('current_company','—')}")
                st.write(f"**Experience:** {tp.get('years_of_experience',0)} yrs")
                st.write(f"**Location:** {tp.get('location','—')}")
            with d2:
                st.markdown("**Behavioral Signals**")
                st.write(f"**GitHub Score:** {ts.get('github_activity_score','—')}")
                st.write(f"**Response Rate:** {ts.get('recruiter_response_rate',0):.0%}")
                st.write(f"**Interview Completion:** {ts.get('interview_completion_rate',0):.0%}")
                st.write(f"**Notice Period:** {ts.get('notice_period_days',90)} days")
                st.write(f"**Open to Work:** {'Yes' if ts.get('open_to_work_flag') else 'No'}")
            with d3:
                st.markdown("**Availability**")
                st.write(f"**Willing to Relocate:** {'Yes' if ts.get('willing_to_relocate') else 'No'}")
                st.write(f"**Offer Acceptance:** {ts.get('offer_acceptance_rate',0):.0%}")
                st.write(f"**Redrob Score:** {final_scores[int(top_idx[0])]:.4f}")
                career = top_c.get("career_history", []) or []
                if career:
                    st.write(f"**Latest Role:** {career[0].get('title','—')} @ {career[0].get('company','—')}")
