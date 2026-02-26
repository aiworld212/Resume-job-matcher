"""
Resume–Job Matching System — Streamlit App
Run: streamlit run app.py
"""

import os
import re
import time
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Resume–Job Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .score-box {
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────
def clean_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).lower().strip()

def score_color(s):
    if s >= 70: return "#27AE60"
    if s >= 50: return "#F39C12"
    return "#E74C3C"

def score_label(s):
    if s >= 70: return "🟢 Excellent Match"
    if s >= 50: return "🟡 Good Match"
    if s >= 35: return "🟠 Fair Match"
    return "🔴 Low Match"

SKILLS = [
    "python", "java", "sql", "excel", "machine learning", "deep learning",
    "nlp", "tensorflow", "pytorch", "pandas", "numpy", "spark", "aws",
    "azure", "docker", "kubernetes", "git", "javascript", "react",
    "tableau", "powerbi", "communication", "leadership", "management",
    "sales", "marketing", "accounting", "finance", "healthcare",
    "design", "research", "c++", "php",
]

def skill_gap(resume_text, job_desc):
    r, j = resume_text.lower(), job_desc.lower()
    required = [s for s in SKILLS if s in j]
    matched  = [s for s in required if s in r]
    missing  = [s for s in required if s not in r]
    pct = round(len(matched) / len(required) * 100, 1) if required else 0
    return {"matched": matched, "missing": missing, "pct": pct}

# ─────────────────────────────────────────────
# Load Model & Data
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data
def load_jobs():
    path = "data/processed/job_descriptions.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    # Fallback built-in job descriptions
    jobs = {
        "INFORMATION-TECHNOLOGY": "Skilled IT professional with software development, Python, Java, SQL, cloud platforms AWS Azure, networking, and cybersecurity experience.",
        "BUSINESS-DEVELOPMENT":   "Business Development Manager to build client relationships, negotiate contracts, drive revenue. Strong CRM and strategic planning skills.",
        "ADVOCATE":               "Legal professional with litigation, contract drafting, legal research, and client counseling experience.",
        "CHEF":                   "Experienced Chef with culinary arts, menu planning, kitchen management, food safety, and team leadership skills.",
        "FINANCE":                "Finance professional with financial analysis, budgeting, forecasting, accounting, Excel, and ERP systems experience.",
        "ENGINEERING":            "Engineer with CAD, project management, quality control, and technical problem-solving skills.",
        "ACCOUNTANT":             "Accountant with bookkeeping, tax preparation, financial statements, auditing, and QuickBooks expertise.",
        "FITNESS":                "Fitness Trainer with personal training certification, nutrition counseling, and program design skills.",
        "AVIATION":               "Aviation professional with pilot license or aviation management, FAA regulations, and flight operations knowledge.",
        "SALES":                  "Sales professional with prospecting, negotiation, CRM Salesforce, and proven track record of exceeding targets.",
        "HEALTHCARE":             "Healthcare professional with clinical knowledge, patient care, EMR systems, and HIPAA compliance experience.",
        "CONSULTANT":             "Consultant with analytical thinking, client management, project delivery, and strong presentation skills.",
        "BANKING":                "Banking professional with financial products knowledge, risk assessment, regulatory compliance, and customer service.",
        "CONSTRUCTION":           "Construction professional with project management, blueprint reading, site supervision, and OSHA compliance.",
        "PUBLIC-RELATIONS":       "PR specialist with media relations, press releases, crisis communication, and social media management.",
        "HR":                     "HR professional with talent acquisition, employee relations, performance management, and labor law compliance.",
        "DESIGNER":               "Designer proficient in Adobe Creative Suite, UI UX, Figma, branding, and responsive design.",
        "ARTS":                   "Arts professional with creative portfolio, exhibition experience, and collaboration skills.",
        "TEACHER":                "Teacher with curriculum development, classroom management, lesson planning, and teaching certification.",
        "APPAREL":                "Apparel professional with fashion design, textile knowledge, trend forecasting, and merchandising skills.",
        "DIGITAL-MEDIA":          "Digital Media specialist with content creation, video editing, SEO, and social media strategy.",
        "AGRICULTURE":            "Agriculture professional with crop science, farm management, irrigation, and sustainable practice knowledge.",
        "AUTOMOBILE":             "Automobile professional with vehicle mechanics, diagnostics, auto repair, and customer service expertise.",
        "BPO":                    "BPO professional with strong communication, customer service, data processing, and call center operations experience.",
    }
    return pd.DataFrame([
        {"job_id": i+1, "job_title": k, "job_description": v}
        for i, (k, v) in enumerate(jobs.items())
    ])

@st.cache_data
def load_resume_data():
    path = "data/raw/Resume.csv"
    if os.path.exists(path):
        df = pd.read_csv(path)
        df["word_count"] = df["Resume_str"].apply(lambda x: len(str(x).split()))
        return df
    return None

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/resume.png", width=70)
    st.title("⚙️ Settings")
    top_k     = st.slider("Top N matches to show", 1, 10, 5)
    show_gap  = st.checkbox("Show Skill Gap Analysis", value=True)
    st.divider()
    st.caption("**Model:** all-MiniLM-L6-v2")
    st.caption("**Embedding dim:** 384")
    st.caption("**Similarity:** Cosine")
    st.divider()
    st.caption("📁 Dataset: Resume.csv")
    st.caption("2484 resumes · 24 categories")

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("🎯 Resume–Job Matcher")
st.markdown("<p style='color:#7F8C8D; margin-top:-10px;'>NLP-powered matching using Sentence Transformers + Cosine Similarity</p>", unsafe_allow_html=True)
st.divider()

# Load everything
model   = load_model()
jobs_df = load_jobs()
raw_df  = load_resume_data()

tab1, tab2, tab3 = st.tabs(["📄 Match My Resume", "📊 Dataset Explorer", "📈 How It Works"])

# ═══════════════════════════════════════════════
# TAB 1 — MATCH RESUME
# ═══════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("📋 Your Resume")
        mode = st.radio("Input method", ["Paste Text", "Upload File"], horizontal=True)

        resume_text = ""

        if mode == "Paste Text":
            resume_text = st.text_area(
                "Paste your resume here",
                height=300,
                placeholder="John Doe\nData Scientist with 3 years experience.\nSkills: Python, ML, NLP, SQL, AWS...\nExperience: Built ML models at ABC Corp."
            )
        else:
            uploaded = st.file_uploader("Upload PDF, DOCX or TXT", type=["pdf", "docx", "txt"])
            if uploaded:
                try:
                    if uploaded.name.endswith(".pdf"):
                        import fitz, tempfile, shutil
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            shutil.copyfileobj(uploaded, tmp)
                        doc = fitz.open(tmp.name)
                        resume_text = "\n".join([p.get_text() for p in doc])
                        os.unlink(tmp.name)
                    elif uploaded.name.endswith(".docx"):
                        import docx, tempfile, shutil
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                            shutil.copyfileobj(uploaded, tmp)
                        doc = docx.Document(tmp.name)
                        resume_text = "\n".join([p.text for p in doc.paragraphs])
                        os.unlink(tmp.name)
                    else:
                        resume_text = uploaded.read().decode("utf-8", errors="ignore")
                    st.success(f"✅ Loaded: {uploaded.name}")
                except Exception as e:
                    st.error(f"Error reading file: {e}")

        if resume_text:
            st.caption(f"📝 {len(resume_text.split())} words | {len(resume_text)} characters")

    with col2:
        st.subheader("💼 Available Jobs")
        st.info(f"**{len(jobs_df)} job descriptions** ready for matching")
        with st.expander("👁️ Preview all job descriptions"):
            st.dataframe(
                jobs_df[["job_title", "job_description"]].assign(
                    job_description=jobs_df["job_description"].str[:80] + "..."
                ),
                use_container_width=True,
                hide_index=True
            )

    st.divider()
    btn = st.button("🚀 Find Best Matching Jobs", use_container_width=True, type="primary")

    if btn:
        if not resume_text.strip():
            st.error("⚠️ Please paste or upload your resume first!")
        else:
            with st.spinner("⚙️ Encoding resume and computing similarity scores..."):
                t0 = time.time()

                cleaned   = clean_text(resume_text)
                r_emb     = model.encode([cleaned], normalize_embeddings=True)
                j_embs    = model.encode(jobs_df["job_description"].tolist(),
                                         normalize_embeddings=True,
                                         show_progress_bar=False)
                scores    = cosine_similarity(r_emb, j_embs)[0]
                elapsed   = time.time() - t0

            top_idx = np.argsort(scores)[::-1][:top_k]
            results = [{
                "Rank":        i + 1,
                "Job Title":   jobs_df.iloc[idx]["job_title"],
                "Match Score": round(scores[idx] * 100, 1),
                "Label":       score_label(scores[idx] * 100),
                "desc":        jobs_df.iloc[idx]["job_description"],
            } for i, idx in enumerate(top_idx)]
            res_df = pd.DataFrame(results)

            st.success(f"✅ Matched against {len(jobs_df)} jobs in {elapsed:.2f}s")

            # ── Score Cards ──────────────────────────────
            st.subheader("🏆 Top Matches")
            cards = st.columns(min(3, len(res_df)))
            for i, row in res_df.head(3).iterrows():
                with cards[i]:
                    st.markdown(
                        f'<div class="score-box" style="background:{score_color(row["Match Score"])}">'
                        f'#{row["Rank"]} {row["Job Title"]}<br>'
                        f'{row["Match Score"]}%<br>'
                        f'<small>{row["Label"]}</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            # ── Bar Chart ────────────────────────────────
            fig = px.bar(
                res_df, x="Match Score", y="Job Title",
                orientation="h",
                color="Match Score",
                color_continuous_scale="RdYlGn",
                range_color=[0, 100],
                text="Match Score",
                title="🎯 Match Scores (%)",
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(
                coloraxis_showscale=False,
                yaxis={"categoryorder": "total ascending"},
                height=350,
                margin=dict(l=10, r=80, t=50, b=10)
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Results Table ────────────────────────────
            st.dataframe(
                res_df[["Rank", "Job Title", "Match Score", "Label"]],
                use_container_width=True,
                hide_index=True
            )

            # ── Skill Gap ────────────────────────────────
            if show_gap:
                st.subheader(f"🧩 Skill Gap — Top Match: {res_df.iloc[0]['Job Title']}")
                gap = skill_gap(resume_text, res_df.iloc[0]["desc"])

                g1, g2 = st.columns(2)
                with g1:
                    fig2 = go.Figure(go.Pie(
                        labels=["Matched", "Missing"],
                        values=[max(1, len(gap["matched"])), max(1, len(gap["missing"]))],
                        hole=0.6,
                        marker_colors=["#27AE60", "#E74C3C"],
                    ))
                    fig2.update_layout(
                        height=280,
                        margin=dict(l=10, r=10, t=10, b=10),
                        annotations=[{
                            "text": f"{gap['pct']}%",
                            "x": 0.5, "y": 0.5,
                            "font_size": 24,
                            "showarrow": False
                        }]
                    )
                    st.plotly_chart(fig2, use_container_width=True)

                with g2:
                    st.markdown("**✅ Skills You Have**")
                    if gap["matched"]:
                        st.success("  ·  ".join(gap["matched"]))
                    else:
                        st.warning("No matching skills detected")

                    st.markdown("**❌ Skills to Add to Resume**")
                    if gap["missing"]:
                        st.error("  ·  ".join(gap["missing"]))
                    else:
                        st.success("🎉 You have all required skills!")

# ═══════════════════════════════════════════════
# TAB 2 — DATASET EXPLORER
# ═══════════════════════════════════════════════
with tab2:
    st.subheader("📊 Resume Dataset Overview")

    if raw_df is not None:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Resumes",  f"{len(raw_df):,}")
        m2.metric("Categories",     raw_df["Category"].nunique())
        m3.metric("Avg Word Count", f"{raw_df['word_count'].mean():.0f}")
        m4.metric("Max Word Count", f"{raw_df['word_count'].max():,}")

        st.divider()

        c1, c2 = st.columns(2)
        with c1:
            cat_df = raw_df["Category"].value_counts().reset_index()
            fig_c  = px.bar(cat_df, x="count", y="Category",
                            orientation="h", title="Resumes per Category",
                            color="count", color_continuous_scale="Blues")
            fig_c.update_layout(coloraxis_showscale=False, height=500)
            st.plotly_chart(fig_c, use_container_width=True)

        with c2:
            fig_w = px.box(raw_df, x="Category", y="word_count",
                           title="Word Count by Category", color="Category")
            fig_w.update_layout(showlegend=False, height=500)
            fig_w.update_xaxes(tickangle=45)
            st.plotly_chart(fig_w, use_container_width=True)

        st.subheader("🔍 Browse Resumes")
        sel = st.selectbox("Filter by Category",
                           ["All"] + sorted(raw_df["Category"].unique().tolist()))
        filtered = raw_df if sel == "All" else raw_df[raw_df["Category"] == sel]
        st.dataframe(filtered[["ID", "Category", "word_count"]].head(20),
                     use_container_width=True, hide_index=True)

        if sel != "All":
            sample = filtered["Resume_str"].iloc[0]
            with st.expander("📄 View sample resume"):
                st.text(sample[:1500] + ("..." if len(sample) > 1500 else ""))
    else:
        st.warning("⚠️ Place Resume.csv in data/raw/ to explore the dataset.")

# ═══════════════════════════════════════════════
# TAB 3 — HOW IT WORKS
# ═══════════════════════════════════════════════
with tab3:
    st.subheader("🧠 How the Matching Works")
    st.markdown("""
    ### Pipeline
    ```
    Your Resume Text
         ↓
    Text Cleaning  →  remove URLs, emails, HTML, special chars
         ↓
    Sentence Transformer (all-MiniLM-L6-v2)
         ↓
    384-dimensional Embedding Vector
         ↓
    Cosine Similarity  ←→  Job Description Embeddings
         ↓
    Ranked Results with Match Score (0–100%)
    ```

    ### Why Sentence Transformers?
    - Understands **meaning**, not just keywords
    - "ML engineer" and "machine learning developer" → high similarity ✅
    - Pure keyword matching would miss this completely

    ### Model Info
    | Property | Value |
    |---|---|
    | Model | all-MiniLM-L6-v2 |
    | Embedding Dim | 384 |
    | Model Size | ~80MB |
    | Speed | ~5ms per sentence |

    ### Score Guide
    | Score | Meaning |
    |---|---|
    | 🟢 70–100% | Excellent Match |
    | 🟡 50–70% | Good Match |
    | 🟠 35–50% | Fair Match |
    | 🔴 0–35% | Low Match |
    """)
