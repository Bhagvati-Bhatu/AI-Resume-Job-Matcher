import streamlit as st
import html

from src.document_loader import extract_text_from_pdf
from src.vector_store import create_vector_store

from src.rag_pipeline import (
    analyze_resume,
    extract_job_skills
)

from src.embeddings import create_embeddings

from src.ats_scorer import (
    calculate_skill_match,
    calculate_structure_score,
    calculate_semantic_similarity,
    calculate_ats_score
)

from src.bullet_optimizer import optimize_bullet


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Resume Matcher",
    page_icon="■",
    layout="wide"
)


# =========================================================
# SESSION STATE
# =========================================================

if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None

if "optimized_bullet" not in st.session_state:
    st.session_state.optimized_bullet = None

if "optimized_original" not in st.session_state:
    st.session_state.optimized_original = None

if "active_section" not in st.session_state:
    st.session_state.active_section = "Overview"


# =========================================================
# CUSTOM CSS — neobrutalist / dev-tool dashboard style
# =========================================================

st.markdown(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', ui-monospace, monospace;
    }

    .stApp, [data-testid="stAppViewContainer"] {
        background: #ECEEE7;
    }

    #MainMenu, footer, header[data-testid="stHeader"] {visibility: hidden;}

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1180px;
    }

    h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: #111111;
    }

    /* ---------------------------------------------------
       TOP BAR
    --------------------------------------------------- */
    .topbar {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        padding-bottom: 14px;
        border-bottom: 2px solid #111111;
        margin-bottom: 20px;
    }

    .topbar-title {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 1.9rem;
        font-weight: 700;
        color: #111111;
        letter-spacing: -0.01em;
        line-height: 1;
    }

    .topbar-caption {
        font-size: 0.72rem;
        font-weight: 600;
        color: #1A46E0;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-top: 6px;
    }

    /* ---------------------------------------------------
       SECTION LABEL — "■ TITLE" pattern
    --------------------------------------------------- */
    .block-label {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        color: #111111;
        margin-bottom: 10px;
    }

    .block-label .sq {
        display: inline-block;
        width: 9px;
        height: 9px;
        background: #1A46E0;
        flex-shrink: 0;
    }

    .comment-label {
        font-size: 0.72rem;
        font-weight: 600;
        color: #1A46E0;
        text-transform: uppercase;
        letter-spacing: 0.02em;
        margin-bottom: 8px;
    }

    /* ---------------------------------------------------
       PANELS / CARDS — sharp corners, black border, no shadow
    --------------------------------------------------- */
    .panel {
        background: #FFFFFF;
        border: 1.5px solid #111111;
        border-radius: 0;
        padding: 16px 18px;
    }

    .box {
        background: #FFFFFF;
        border: 1.5px solid #111111;
        border-radius: 0;
        padding: 12px 14px;
        margin-bottom: 12px;
        font-size: 0.84rem;
        color: #333029;
        line-height: 1.55;
    }

    .file-status {
        font-size: 0.78rem;
        font-weight: 600;
        color: #3F8F5F;
        margin-top: 8px;
    }

    .word-count {
        font-size: 0.72rem;
        color: #8A887F;
        margin-top: 6px;
    }

    /* ---------------------------------------------------
       EMPTY STATE
    --------------------------------------------------- */
    .empty-wrap {
        border-top: 1.5px solid #111111;
        padding-top: 18px;
        margin-top: 8px;
    }

    .steps-line {
        display: flex;
        gap: 26px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #333029;
        text-transform: uppercase;
        margin-top: 10px;
    }

    .steps-line b {
        color: #1A46E0;
        margin-right: 6px;
    }

    /* ---------------------------------------------------
       STATS BAR
    --------------------------------------------------- */
    .stats-bar {
        display: flex;
        gap: 28px;
        flex-wrap: wrap;
        border: 1.5px solid #111111;
        background: #FFFFFF;
        padding: 10px 16px;
        margin: 14px 0 16px 0;
        font-size: 0.76rem;
        font-weight: 600;
        color: #333029;
        text-transform: uppercase;
    }

    .stats-bar b { color: #1A46E0; }

    /* ---------------------------------------------------
       SCORE BLOCK
    --------------------------------------------------- */
    .score-wrap {
        border-top: 1.5px solid #111111;
        padding-top: 16px;
        margin-top: 4px;
    }

    .score-row {
        display: flex;
        align-items: baseline;
        gap: 10px;
    }

    .score-value {
        font-family: Georgia, 'Times New Roman', serif;
        font-size: 2.3rem;
        font-weight: 700;
        color: #111111;
        line-height: 1;
    }

    .score-max {
        font-size: 1rem;
        color: #8A887F;
        font-weight: 500;
    }

    .score-band {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-left: 6px;
    }

    .score-track {
        width: 100%;
        max-width: 460px;
        height: 8px;
        background: #FFFFFF;
        border: 1.5px solid #111111;
        margin-top: 10px;
    }

    .score-fill {
        height: 100%;
    }

    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 10px;
        margin-top: 16px;
    }

    @media (max-width: 720px) {
        .metrics-grid { grid-template-columns: repeat(2, 1fr); }
    }

    .metric-box {
        border: 1.5px solid #111111;
        background: #FFFFFF;
        padding: 8px 10px;
    }

    .metric-label {
        font-size: 0.68rem;
        color: #8A887F;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 4px;
    }

    .metric-value {
        font-size: 1rem;
        font-weight: 700;
        color: #1A46E0;
    }

    /* ---------------------------------------------------
       SECTION NAV — bordered box buttons
    --------------------------------------------------- */
    div[class*="st-key-section_nav"] {
        border: 1.5px solid #111111;
        border-bottom: none;
        display: flex;
    }

    div[class*="st-key-section_nav"] button {
        border: none !important;
        border-right: 1.5px solid #111111 !important;
        border-radius: 0 !important;
        box-shadow: none !important;
        font-size: 0.76rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        padding: 8px 4px !important;
    }

    div[class*="st-key-section_nav"] button[kind="secondary"] {
        background: #FFFFFF !important;
        color: #8A887F !important;
    }

    div[class*="st-key-section_nav"] button[kind="secondary"]:hover {
        color: #111111 !important;
        background: #F3F2EC !important;
    }

    div[class*="st-key-section_nav"] button[kind="primary"] {
        background: #1A46E0 !important;
        color: #FFFFFF !important;
    }

    .section-body {
        border: 1.5px solid #111111;
        border-top: none;
        padding: 18px 20px;
        background: #FFFFFF;
        margin-bottom: 20px;
    }

    /* ---------------------------------------------------
       TAGS — keyword-highlight motif
    --------------------------------------------------- */
    .tag-row {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin: 4px 0 14px 0;
    }

    .tag {
        display: inline-block;
        border: 1.3px solid #111111;
        padding: 3px 9px;
        font-size: 0.76rem;
        font-weight: 600;
        border-radius: 0;
    }

    .tag-neutral { background: #FFFFFF; color: #333029; }
    .tag-match { background: #FFE270; color: #4A3B00; }
    .tag-missing { background: #F7C9C9; color: #7A1F1F; }

    /* ---------------------------------------------------
       INDICATOR ITEMS
    --------------------------------------------------- */
    .indicator-item {
        border: 1.3px solid #111111;
        border-left-width: 5px;
        padding: 8px 12px;
        margin-bottom: 8px;
        font-size: 0.82rem;
        color: #333029;
        line-height: 1.45;
        background: #FFFFFF;
    }

    .indicator-issue { border-left-color: #D97706; }
    .indicator-rec { border-left-color: #1A46E0; }
    .indicator-high { border-left-color: #C0392B; font-weight: 700; }
    .indicator-medium { border-left-color: #D97706; }
    .indicator-low { border-left-color: #8A887F; }

    .priority-heading {
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin: 14px 0 6px 0;
        color: #8A887F;
    }

    /* ---------------------------------------------------
       BUTTONS
    --------------------------------------------------- */
    /* Keep every Streamlit primary/blue button readable. */
    button[kind="primary"],
    button[data-testid="baseButton-primary"] {
        color: #FFFFFF !important;
    }

    button[kind="primary"] *,
    button[data-testid="baseButton-primary"] * {
        color: #FFFFFF !important;
    }

    div[class*="st-key-cta"] button[kind="primary"] {
        background: #1A46E0 !important;
        border: 1.5px solid #111111 !important;
        border-radius: 0 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 0.84rem !important;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        padding: 0.55rem 1rem !important;
        box-shadow: none !important;
    }

    div[class*="st-key-cta"] button[kind="primary"]:hover {
        background: #123AC4 !important;
        color: #FFFFFF !important;
    }

    /* ---------------------------------------------------
       INPUTS — force light theme, sharp corners
    --------------------------------------------------- */
    div[data-testid="stFileUploader"] section {
        border-radius: 0;
        border: 1.5px dashed #111111;
        background: #FFFFFF;
        padding: 12px;
    }

    div[data-testid="stFileUploader"] section span,
    div[data-testid="stFileUploader"] section small {
        color: #55534B !important;
    }

    /* Upload/Browse button: blue background with white text. */
    div[data-testid="stFileUploader"] button {
        background: #1A46E0 !important;
        border: 1.5px solid #111111 !important;
        border-radius: 0 !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: none !important;
    }

    div[data-testid="stFileUploader"] button:hover {
        background: #123AC4 !important;
        color: #FFFFFF !important;
    }

    div[data-testid="stFileUploader"] button * {
        color: #FFFFFF !important;
    }

    .stTextArea textarea,
    .stTextInput input {
        background-color: #FFFFFF !important;
        color: #111111 !important;
        border: 1.5px solid #111111 !important;
        border-radius: 0 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }

    .stTextArea textarea::placeholder {
        color: #A8A69C !important;
    }

    /* ---------------------------------------------------
       EXPANDER (RAG evidence)
    --------------------------------------------------- */
    div[data-testid="stExpander"] {
        border: 1.5px solid #111111 !important;
        border-radius: 0 !important;
        background: #FFFFFF !important;
        margin-bottom: 8px;
    }

    div[data-testid="stExpander"] summary {
        font-size: 0.8rem !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        color: #111111 !important;
    }

    /* ---------------------------------------------------
       FOOTER
    --------------------------------------------------- */
    .site-footer {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 24px;
        padding-top: 12px;
        border-top: 1.5px solid #111111;
        font-size: 0.72rem;
        font-weight: 600;
        color: #8A887F;
        text-transform: uppercase;
    }

    .site-footer b { color: #111111; }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# HELPER FUNCTIONS (presentation only — no scoring logic here)
# =========================================================

def get_score_meta(score):
    """Return (label, color, description) for a 0-100 score band."""

    if score >= 80:
        return ("Excellent match", "#3F8F5F", "Your resume is strongly aligned with this role.")
    elif score >= 60:
        return ("Good match", "#1A46E0", "Your resume aligns well with this role, with a few areas to strengthen.")
    elif score >= 40:
        return ("Moderate match", "#D97706", "Your resume partially aligns with this role — some key gaps to address.")
    else:
        return ("Low match", "#C0392B", "Your resume needs meaningful updates to align with this role.")


def render_topbar():

    st.markdown(
        """
        <div class="topbar">
            <div>
                <div class="topbar-title">Resume Matcher</div>
                <div class="topbar-caption">// ATS Compatibility Engine</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_input_workspace():
    """Renders the upload + job description panels and the Analyze button.
    Returns (resume_file, job_description, analyze_clicked)."""

    col1, col2 = st.columns(2)

    with col1:

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="block-label"><span class="sq"></span>Your Resume</div>', unsafe_allow_html=True)

        resume_file = st.file_uploader(
            "Upload Resume",
            type=["pdf"],
            help="Upload your resume in PDF format.",
            label_visibility="collapsed"
        )

        if resume_file:
            st.markdown(
                f'<div class="file-status">✓ {html.escape(resume_file.name)}</div>',
                unsafe_allow_html=True
            )

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="block-label"><span class="sq"></span>Job Description</div>', unsafe_allow_html=True)

        job_description = st.text_area(
            "Job Description",
            height=132,
            placeholder="Paste the job description here...",
            label_visibility="collapsed"
        )

        word_count = len(job_description.split()) if job_description else 0
        st.markdown(f'<div class="word-count">{word_count} words</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")

    with st.container(key="cta_analyze"):

        btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])

        with btn_col2:
            analyze_clicked = st.button(
                "Analyze Resume →",
                type="primary",
                use_container_width=True
            )

    return resume_file, job_description, analyze_clicked


def render_empty_state():

    st.markdown(
        """
        <div class="empty-wrap">
            <div class="comment-label">// No analysis yet</div>
            <div class="steps-line">
                <span><b>01</b>UPLOAD</span>
                <span><b>02</b>COMPARE</span>
                <span><b>03</b>IMPROVE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_stats_bar(required_skills, skill_result, ats_score):

    st.markdown(
        f"""
        <div class="stats-bar">
            <div><b>{len(required_skills)}</b> skills detected</div>
            <div><b>{len(skill_result["matched_skills"])}</b> matches found</div>
            <div>Match rate: <b>{ats_score}%</b></div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_score_block(ats_score, skill_score, semantic_score, experience_score, structure_score):

    label, color, _ = get_score_meta(ats_score)

    metrics = [
        ("Skills", skill_score, "%"),
        ("Semantic", semantic_score, "%"),
        ("Experience", experience_score, "%"),
        ("Structure", structure_score, "/10"),
    ]

    boxes = "".join(
        f"""<div class="metric-box">
            <div class="metric-label">{html.escape(lbl)}</div>
            <div class="metric-value">{val}{suffix}</div>
        </div>"""
        for lbl, val, suffix in metrics
    )

    st.markdown(
        f"""
        <div class="score-wrap">
            <div class="block-label"><span class="sq"></span>Resume Match Score</div>
            <div class="score-row">
                <span class="score-value">{ats_score}</span>
                <span class="score-max">/ 100</span>
                <span class="score-band" style="color:{color};">{html.escape(label)}</span>
            </div>
            <div class="score-track"><div class="score-fill" style="width:{ats_score}%;background:{color};"></div></div>
            <div class="metrics-grid">{boxes}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_section_nav():

    sections = ["Overview", "Skills", "ATS Insights", "RAG Evidence", "Bullet Optimizer"]

    with st.container(key="section_nav"):

        cols = st.columns(len(sections))

        for col, section in zip(cols, sections):

            with col:
                is_active = st.session_state.active_section == section

                if st.button(
                    section,
                    key=f"nav_{section}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True
                ):
                    st.session_state.active_section = section
                    st.rerun()


def render_tags(items, kind="tag-neutral", empty_message="None found."):

    if not items:
        st.markdown(f'<p style="font-size:0.8rem;color:#8A887F;">{empty_message}</p>', unsafe_allow_html=True)
        return

    tags_html = "".join(
        f'<span class="tag {kind}">{html.escape(str(item))}</span>'
        for item in items
    )

    st.markdown(f'<div class="tag-row">{tags_html}</div>', unsafe_allow_html=True)


def render_indicator(text, kind):

    st.markdown(f'<div class="indicator-item indicator-{kind}">{text}</div>', unsafe_allow_html=True)


def render_footer():

    st.markdown(
        """
        <div class="site-footer">
            <div>✱ <b>Resume Matcher</b></div>
            <div>RAG &middot; FAISS &middot; LangChain &middot; Hugging Face</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# TOP BAR
# =========================================================

render_topbar()


# =========================================================
# INPUT WORKSPACE
# =========================================================

resume_file, job_description, analyze_button = render_input_workspace()


# =========================================================
# ANALYSIS
# =========================================================

if analyze_button:

    if resume_file is None:

        st.error("Please upload your resume PDF.")

    elif not job_description.strip():

        st.error("Please enter a job description.")

    else:

        with st.spinner("Running AI-powered ATS analysis..."):

            # ---------------------------------------------
            # SAVE RESUME
            # ---------------------------------------------

            resume_path = "temp_resume.pdf"

            with open(resume_path, "wb") as file:
                file.write(resume_file.getbuffer())

            # ---------------------------------------------
            # EXTRACT RESUME TEXT
            # ---------------------------------------------

            resume_text = extract_text_from_pdf(resume_path)

            # ---------------------------------------------
            # CREATE VECTOR STORE
            # ---------------------------------------------

            vector_store = create_vector_store(resume_text)

            # ---------------------------------------------
            # RAG RETRIEVAL
            # ---------------------------------------------

            results = vector_store.similarity_search(
                job_description,
                k=6
            )

            # ---------------------------------------------
            # COMBINE RETRIEVED RESUME SECTIONS
            # ---------------------------------------------

            resume_sections = "\n\n".join(
                result.page_content
                for result in results
            )

            # ---------------------------------------------
            # LLM RESUME ANALYSIS
            # ---------------------------------------------

            analysis = analyze_resume(
                job_description=job_description,
                resume_sections=resume_sections
            )

            # ---------------------------------------------
            # AI JOB SKILL EXTRACTION
            # ---------------------------------------------

            required_skills = extract_job_skills(job_description)

            # ---------------------------------------------
            # CREATE EMBEDDINGS
            # ---------------------------------------------

            embeddings = create_embeddings()

            # ---------------------------------------------
            # SKILL MATCH
            # ---------------------------------------------

            skill_result = calculate_skill_match(
                required_skills=required_skills,
                resume_text=resume_text,
                embeddings=embeddings
            )

            skill_score = skill_result["score"]

            # ---------------------------------------------
            # SEMANTIC MATCH
            # ---------------------------------------------

            chunk_scores = []

            for result in results:

                score = calculate_semantic_similarity(
                    embeddings=embeddings,
                    job_description=job_description,
                    resume_text=result.page_content
                )

                chunk_scores.append(score)

            semantic_score = max(chunk_scores) if chunk_scores else 0

            # ---------------------------------------------
            # OTHER SCORES
            # ---------------------------------------------

            experience_score = analysis.experience_match

            structure_score = calculate_structure_score(resume_text)

            # ---------------------------------------------
            # FINAL ATS SCORE
            # ---------------------------------------------

            ats_score = calculate_ats_score(
                skill_score=skill_score,
                semantic_score=semantic_score,
                experience_score=experience_score,
                structure_score=structure_score
            )

        st.session_state.analysis_data = {
            "ats_score": ats_score,
            "skill_score": skill_score,
            "semantic_score": semantic_score,
            "experience_score": experience_score,
            "structure_score": structure_score,
            "skill_result": skill_result,
            "required_skills": required_skills,
            "analysis": analysis,
            "rag_sections": [result.page_content for result in results],
            "job_description": job_description,
        }

        st.session_state.optimized_bullet = None
        st.session_state.optimized_original = None
        st.session_state.active_section = "Overview"


# =========================================================
# RESULTS
# =========================================================

data = st.session_state.analysis_data

if data is None:

    render_empty_state()

else:

    ats_score = data["ats_score"]
    skill_score = data["skill_score"]
    semantic_score = data["semantic_score"]
    experience_score = data["experience_score"]
    structure_score = data["structure_score"]
    skill_result = data["skill_result"]
    required_skills = data["required_skills"]
    analysis = data["analysis"]
    rag_sections = data["rag_sections"]
    job_description_for_bullet = data["job_description"]

    render_stats_bar(required_skills, skill_result, ats_score)

    render_score_block(ats_score, skill_score, semantic_score, experience_score, structure_score)

    st.write("")

    render_section_nav()

    st.markdown('<div class="section-body">', unsafe_allow_html=True)

    active = st.session_state.active_section

    # -----------------------------------------------------
    # OVERVIEW
    # -----------------------------------------------------

    if active == "Overview":

        matched_count = len(skill_result["matched_skills"])
        total_count = len(required_skills)

        st.markdown('<div class="block-label"><span class="sq"></span>Match Summary</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="box">Your resume matched {matched_count} of {total_count} skills '
            f'identified in the job description, with an overall ATS compatibility score of '
            f'{ats_score}/100.</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="block-label"><span class="sq"></span>Strong Matches</div>', unsafe_allow_html=True)
            render_tags(skill_result["matched_skills"][:8], "tag-match", "No strong matches identified yet.")

        with col2:
            st.markdown('<div class="block-label"><span class="sq"></span>Areas To Improve</div>', unsafe_allow_html=True)
            attention_items = skill_result["missing_skills"][:5] + analysis.ats_issues[:2]
            render_tags(attention_items, "tag-missing", "No major gaps identified.")

    # -----------------------------------------------------
    # SKILLS
    # -----------------------------------------------------

    elif active == "Skills":

        st.markdown('<div class="block-label"><span class="sq"></span>Skills In This Role</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.78rem;color:#8A887F;margin-top:-6px;">'
            'Technical skills identified from the job description using generative AI.</p>',
            unsafe_allow_html=True
        )
        render_tags(required_skills, "tag-neutral", "No technical skills were extracted from the job description.")

        st.markdown('<div class="block-label" style="margin-top:14px;"><span class="sq"></span>Matched</div>', unsafe_allow_html=True)
        render_tags(skill_result["matched_skills"], "tag-match", "No matching skills found.")

        st.markdown('<div class="block-label" style="margin-top:14px;"><span class="sq"></span>Missing</div>', unsafe_allow_html=True)
        render_tags(skill_result["missing_skills"], "tag-missing", "No major missing skills found.")

    # -----------------------------------------------------
    # ATS INSIGHTS
    # -----------------------------------------------------

    elif active == "ATS Insights":

        st.markdown('<div class="block-label"><span class="sq"></span>ATS Issues</div>', unsafe_allow_html=True)

        if analysis.ats_issues:
            for issue in analysis.ats_issues:
                render_indicator(html.escape(str(issue)), "issue")
        else:
            st.markdown('<p style="font-size:0.8rem;color:#8A887F;">No major ATS issues detected.</p>', unsafe_allow_html=True)

        st.markdown('<div class="block-label" style="margin-top:14px;"><span class="sq"></span>Recommended Changes</div>', unsafe_allow_html=True)

        if analysis.recommendations:
            for recommendation in analysis.recommendations:
                render_indicator(html.escape(str(recommendation)), "rec")
        else:
            st.markdown('<p style="font-size:0.8rem;color:#8A887F;">No recommendations.</p>', unsafe_allow_html=True)

        st.markdown('<div class="block-label" style="margin-top:14px;"><span class="sq"></span>Improvement Plan</div>', unsafe_allow_html=True)

        st.markdown('<div class="priority-heading">High Priority</div>', unsafe_allow_html=True)

        high_priority = [
            f"Consider adding or strengthening <b>{html.escape(str(skill))}</b> "
            "if you have genuine experience with it."
            for skill in skill_result["missing_skills"][:3]
        ]

        if high_priority:
            for item in high_priority:
                render_indicator(item, "high")
        else:
            st.markdown('<p style="font-size:0.8rem;color:#8A887F;">No high-priority gaps identified.</p>', unsafe_allow_html=True)

        st.markdown('<div class="priority-heading">Medium Priority</div>', unsafe_allow_html=True)

        medium_priority = analysis.recommendations[:3]

        if medium_priority:
            for item in medium_priority:
                render_indicator(html.escape(str(item)), "medium")
        else:
            st.markdown('<p style="font-size:0.8rem;color:#8A887F;">No medium-priority improvements identified.</p>', unsafe_allow_html=True)

        st.markdown('<div class="priority-heading">Low Priority</div>', unsafe_allow_html=True)

        low_priority = []

        if structure_score < 8:
            low_priority.append(
                "Improve resume structure by ensuring important sections such "
                "as Skills, Projects, Education, and Experience are clearly organized."
            )

        if semantic_score < 60:
            low_priority.append(
                "Improve keyword and content alignment between your resume "
                "and the target job description."
            )

        if low_priority:
            for item in low_priority:
                render_indicator(item, "low")
        else:
            st.markdown('<p style="font-size:0.8rem;color:#8A887F;">No major low-priority improvements identified.</p>', unsafe_allow_html=True)

    # -----------------------------------------------------
    # RAG EVIDENCE
    # -----------------------------------------------------

    elif active == "RAG Evidence":

        st.markdown('<div class="block-label"><span class="sq"></span>Retrieved Resume Context</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.78rem;color:#8A887F;margin-top:-6px;">'
            'These are the resume sections retrieved through semantic search and '
            'provided to the analysis model.</p>',
            unsafe_allow_html=True
        )

        for i, section in enumerate(rag_sections, start=1):

            with st.expander(f"Resume Context {i:02d}"):
                st.write(section)

    # -----------------------------------------------------
    # BULLET OPTIMIZER
    # -----------------------------------------------------

    elif active == "Bullet Optimizer":

        st.markdown('<div class="block-label"><span class="sq"></span>Rewrite A Resume Bullet</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-size:0.78rem;color:#8A887F;margin-top:-6px;">'
            'Make a resume bullet clearer and more relevant to the role.</p>',
            unsafe_allow_html=True
        )

        bullet = st.text_area(
            "Original bullet",
            placeholder="Example: Developed a machine learning model using Python.",
            key="bullet_input"
        )

        with st.container(key="cta_bullet"):

            btn_col1, btn_col2 = st.columns([3, 1])

            with btn_col2:
                optimize_clicked = st.button(
                    "Improve Bullet",
                    type="primary",
                    use_container_width=True
                )

        if optimize_clicked:

            if not bullet.strip():

                st.warning("Please enter a resume bullet.")

            else:

                with st.spinner("Optimizing resume bullet..."):

                    optimized_bullet = optimize_bullet(
                        resume_bullet=bullet,
                        job_description=job_description_for_bullet
                    )

                st.session_state.optimized_bullet = optimized_bullet
                st.session_state.optimized_original = bullet

        if st.session_state.optimized_bullet:

            st.write("")

            result_col1, result_col2 = st.columns(2)

            with result_col1:
                st.markdown('<div class="block-label"><span class="sq"></span>Before</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="box">{html.escape(str(st.session_state.optimized_original))}</div>',
                    unsafe_allow_html=True
                )

            with result_col2:
                st.markdown('<div class="block-label"><span class="sq"></span>After</div>', unsafe_allow_html=True)
                st.code(st.session_state.optimized_bullet, language=None)

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

render_footer()