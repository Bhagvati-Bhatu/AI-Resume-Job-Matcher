import streamlit as st
import html
import textwrap

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
    page_title="Resume Match",
    page_icon="▪",
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
# DESIGN SYSTEM
# =========================================================

st.html(
    textwrap.dedent(
        """
    <style>

    /* -----------------------------------------------------
       GLOBAL
    ----------------------------------------------------- */

    :root {
        --paper: #F3F1E7;
        --paper-2: #F8F7F1;
        --ink: #171717;
        --muted: #6B6B63;
        --line: #262626;
        --soft-line: #D8D5C9;
        --blue: #2457E6;
        --blue-dark: #173FAE;
        --green: #247A43;
        --orange: #A96516;
        --red: #A3312B;
    }

    html, body, [class*="css"] {
        font-family: Arial, Helvetica, sans-serif;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: var(--paper) !important;
    }

    #MainMenu,
    footer,
    header[data-testid="stHeader"] {
        visibility: hidden;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 22px;
        padding-bottom: 40px;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: Georgia, "Times New Roman", serif !important;
        color: var(--ink) !important;
    }

    /* Avoid the previous global p/span/div color rule.
       It could interfere with Streamlit's own widgets. */

    /* -----------------------------------------------------
       TOP BAR
    ----------------------------------------------------- */

    .topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        border-bottom: 1px solid var(--line);
        padding: 5px 0 11px 0;
        margin-bottom: 34px;
    }

    .brand {
        color: var(--ink);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 18px;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .top-links {
        display: flex;
        gap: 24px;
        color: var(--muted);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    /* -----------------------------------------------------
       PAGE INTRO
    ----------------------------------------------------- */

    .eyebrow {
        color: var(--blue);
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .page-title {
        font-family: Georgia, "Times New Roman", serif;
        font-size: clamp(2.3rem, 5vw, 4.25rem);
        line-height: 0.96;
        letter-spacing: -0.055em;
        color: var(--ink);
        margin: 0 0 14px 0;
    }

    .page-description {
        max-width: 670px;
        color: #44443F;
        font-size: 15px;
        line-height: 1.55;
        margin-bottom: 34px;
    }

    /* -----------------------------------------------------
       INPUT WORKSPACE
    ----------------------------------------------------- */

    .workspace-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0;
        border-top: 1px solid var(--line);
        border-left: 1px solid var(--line);
        margin-bottom: 18px;
    }

    .workspace-cell {
        min-height: 245px;
        padding: 16px;
        border-right: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        background: var(--paper-2);
    }

    .cell-label {
        color: var(--ink);
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.13em;
        text-transform: uppercase;
        margin-bottom: 13px;
    }

    .cell-note {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.45;
        margin-top: 8px;
    }

    .file-name {
        color: var(--green);
        font-size: 12px;
        margin-top: 9px;
        font-weight: 700;
    }

    .word-count {
        color: var(--muted);
        font-size: 11px;
        margin-top: 5px;
    }

    /* File uploader */

    div[data-testid="stFileUploader"] section {
        background: var(--paper-2) !important;
        border: 1px dashed #77776F !important;
        border-radius: 0 !important;
        min-height: 170px;
        padding: 20px !important;
    }

    div[data-testid="stFileUploader"] section span,
    div[data-testid="stFileUploader"] section small {
        color: var(--muted) !important;
    }

    div[data-testid="stFileUploader"] section button {
        border-radius: 0 !important;
        border: 1px solid var(--ink) !important;
        background: var(--paper-2) !important;
        color: var(--ink) !important;
        box-shadow: none !important;
    }

    /* Text area */

    .stTextArea textarea {
        background: var(--paper-2) !important;
        color: var(--ink) !important;
        border: 1px solid #77776F !important;
        border-radius: 0 !important;
        min-height: 170px !important;
        font-size: 13px !important;
        line-height: 1.55 !important;
        box-shadow: none !important;
    }

    .stTextArea textarea:focus {
        border: 1px solid var(--blue) !important;
        box-shadow: 0 0 0 1px var(--blue) !important;
    }

    .stTextArea textarea::placeholder {
        color: #99978E !important;
    }

    /* -----------------------------------------------------
       BUTTONS
    ----------------------------------------------------- */

    .analyze-wrap {
        display: flex;
        justify-content: flex-end;
        margin: 4px 0 42px 0;
    }

    div[class*="st-key-cta_analyze"] button {
        border-radius: 0 !important;
        background: var(--blue) !important;
        border: 1px solid var(--blue) !important;
        color: white !important;
        font-size: 11px !important;
        font-weight: 800 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        min-height: 40px !important;
        box-shadow: none !important;
    }

    div[class*="st-key-cta_analyze"] button:hover {
        background: var(--blue-dark) !important;
        border-color: var(--blue-dark) !important;
        color: white !important;
    }

    div[class*="st-key-cta_bullet"] button {
        border-radius: 0 !important;
        background: var(--blue) !important;
        border: 1px solid var(--blue) !important;
        color: white !important;
        font-size: 11px !important;
        font-weight: 800 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        box-shadow: none !important;
    }

    /* -----------------------------------------------------
       RESULT HEADER
    ----------------------------------------------------- */

    .analysis-rule {
        border-top: 1px solid var(--line);
        margin: 0 0 17px 0;
    }

    .analysis-head {
        display: flex;
        justify-content: space-between;
        align-items: flex-end;
        gap: 20px;
        margin-bottom: 24px;
    }

    .analysis-title {
        font-family: Georgia, "Times New Roman", serif;
        color: var(--ink);
        font-size: 30px;
        letter-spacing: -0.035em;
        margin: 0;
    }

    .analysis-status {
        color: var(--green);
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        white-space: nowrap;
    }

    /* -----------------------------------------------------
       SCORE
    ----------------------------------------------------- */

    .score-layout {
        display: grid;
        grid-template-columns: 230px 1fr;
        gap: 42px;
        border-bottom: 1px solid var(--line);
        padding-bottom: 28px;
    }

    .score-number {
        font-family: Georgia, "Times New Roman", serif;
        color: var(--ink);
        font-size: 68px;
        line-height: 0.88;
        letter-spacing: -0.06em;
        margin-bottom: 10px;
    }

    .score-denom {
        color: var(--muted);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 18px;
    }

    .score-label {
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .score-copy {
        color: #55554F;
        font-size: 12px;
        line-height: 1.5;
        max-width: 420px;
        margin-bottom: 16px;
    }

    .score-track {
        height: 8px;
        background: #D9D7CE;
        width: 100%;
        max-width: 560px;
        border: 1px solid #BDBAAF;
    }

    .score-fill {
        height: 100%;
    }

    /* -----------------------------------------------------
       METRICS
    ----------------------------------------------------- */

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        border-left: 1px solid var(--line);
        border-top: 1px solid var(--line);
        margin-top: 24px;
        margin-bottom: 28px;
    }

    .metric {
        padding: 13px 14px 15px 14px;
        border-right: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        background: var(--paper-2);
    }

    .metric-label {
        color: var(--muted);
        font-size: 9px;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .metric-value {
        color: var(--ink);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 23px;
        margin-bottom: 8px;
    }

    .metric-bar {
        height: 3px;
        background: #D9D7CE;
    }

    .metric-bar-fill {
        height: 100%;
    }

    /* -----------------------------------------------------
       NAVIGATION
    ----------------------------------------------------- */

    div[class*="st-key-section_nav"] {
        border-top: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        margin-top: 4px;
        margin-bottom: 30px;
        padding: 0 !important;
    }

    div[class*="st-key-section_nav"] button {
        min-height: 38px !important;
        border-radius: 0 !important;
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        color: var(--muted) !important;
        font-size: 10px !important;
        font-weight: 800 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        padding: 6px 2px !important;
    }

    div[class*="st-key-section_nav"] button:hover {
        color: var(--ink) !important;
        background: transparent !important;
    }

    div[class*="st-key-section_nav"] button[kind="primary"] {
        color: var(--blue) !important;
        border-bottom: 3px solid var(--blue) !important;
    }

    div[class*="st-key-section_nav"] button[kind="secondary"] {
        border-bottom: 3px solid transparent !important;
    }

    /* -----------------------------------------------------
       RESULT CONTENT
    ----------------------------------------------------- */

    .section-kicker {
        color: var(--blue);
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 7px;
    }

    .section-title {
        color: var(--ink);
        font-family: Georgia, "Times New Roman", serif;
        font-size: 26px;
        letter-spacing: -0.035em;
        margin-bottom: 6px;
    }

    .section-description {
        color: var(--muted);
        font-size: 12px;
        line-height: 1.5;
        margin-bottom: 22px;
    }

    .content-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        border-top: 1px solid var(--line);
        border-left: 1px solid var(--line);
    }

    .content-cell {
        border-right: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        padding: 16px;
        background: var(--paper-2);
        min-height: 130px;
    }

    .content-label {
        color: var(--ink);
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        margin-bottom: 13px;
    }

    .plain-copy {
        color: #3F3F3A;
        font-size: 13px;
        line-height: 1.55;
    }

    .skill-line {
        color: var(--ink);
        font-size: 13px;
        line-height: 1.8;
    }

    .skill-line::before {
        content: "— ";
        color: var(--blue);
        font-weight: 700;
    }

    .muted-empty {
        color: #96948B;
        font-size: 12px;
    }

    /* -----------------------------------------------------
       INDICATORS
    ----------------------------------------------------- */

    .indicator {
        border-top: 1px solid var(--soft-line);
        padding: 11px 0 10px 13px;
        color: #3F3F3A;
        font-size: 12px;
        line-height: 1.5;
        border-left: 3px solid var(--soft-line);
        margin-bottom: 1px;
    }

    .indicator.issue {
        border-left-color: var(--orange);
    }

    .indicator.recommendation {
        border-left-color: var(--blue);
    }

    .indicator.high {
        border-left-color: var(--red);
    }

    .indicator.medium {
        border-left-color: var(--orange);
    }

    .indicator.low {
        border-left-color: #8D8B82;
    }

    .priority-label {
        color: var(--muted);
        font-size: 9px;
        font-weight: 800;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin: 24px 0 5px 0;
    }

    /* -----------------------------------------------------
       RAG EVIDENCE
    ----------------------------------------------------- */

    .rag-card {
        border: 1px solid var(--line);
        background: var(--paper-2);
        margin-bottom: 12px;
    }

    .rag-card-head {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid var(--soft-line);
        padding: 10px 12px;
    }

    .rag-card-title {
        color: var(--ink);
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }

    .rag-card-meta {
        color: var(--muted);
        font-size: 10px;
    }

    .rag-text {
        color: #2E2E2A !important;
        background: var(--paper-2);
        padding: 14px;
        font-family: Arial, Helvetica, sans-serif;
        font-size: 12px;
        line-height: 1.65;
        white-space: pre-wrap;
        overflow-wrap: anywhere;
        min-height: 60px;
    }

    /* -----------------------------------------------------
       BULLET OPTIMIZER
    ----------------------------------------------------- */

    .bullet-label {
        color: var(--ink);
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .before-after {
        display: grid;
        grid-template-columns: 1fr 1fr;
        border-top: 1px solid var(--line);
        border-left: 1px solid var(--line);
        margin-top: 22px;
    }

    .bullet-result {
        border-right: 1px solid var(--line);
        border-bottom: 1px solid var(--line);
        background: var(--paper-2);
        padding: 15px;
        min-height: 145px;
    }

    .bullet-result-label {
        color: var(--muted);
        font-size: 9px;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 13px;
    }

    .bullet-result-text {
        color: #2E2E2A;
        font-size: 13px;
        line-height: 1.6;
        white-space: pre-wrap;
    }

    /* Code block generated by Streamlit */
    div[data-testid="stCode"] {
        border-radius: 0 !important;
        border: 1px solid var(--soft-line) !important;
    }

    /* -----------------------------------------------------
       EMPTY STATE / FOOTER
    ----------------------------------------------------- */

    .empty-state {
        border-top: 1px solid var(--line);
        margin-top: 8px;
        padding: 18px 0 5px 0;
    }

    .empty-title {
        font-family: Georgia, "Times New Roman", serif;
        color: var(--ink);
        font-size: 25px;
        margin-bottom: 5px;
    }

    .empty-copy {
        color: var(--muted);
        font-size: 12px;
        margin-bottom: 18px;
    }

    .steps {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        border-top: 1px solid var(--soft-line);
        border-left: 1px solid var(--soft-line);
        max-width: 700px;
    }

    .step {
        border-right: 1px solid var(--soft-line);
        border-bottom: 1px solid var(--soft-line);
        padding: 12px;
        background: var(--paper-2);
    }

    .step-no {
        color: var(--blue);
        font-size: 10px;
        font-weight: 800;
        margin-bottom: 6px;
    }

    .step-text {
        color: var(--ink);
        font-size: 12px;
        font-weight: 700;
    }

    .footer {
        border-top: 1px solid var(--line);
        margin-top: 55px;
        padding-top: 12px;
        color: var(--muted);
        font-size: 10px;
        letter-spacing: 0.05em;
    }

    /* -----------------------------------------------------
       RESPONSIVE
    ----------------------------------------------------- */

    @media (max-width: 800px) {
        .workspace-grid,
        .content-grid,
        .before-after {
            grid-template-columns: 1fr;
        }

        .score-layout {
            grid-template-columns: 1fr;
            gap: 18px;
        }

        .metric-grid {
            grid-template-columns: repeat(2, 1fr);
        }

        .steps {
            grid-template-columns: 1fr;
        }

        .top-links {
            display: none;
        }

        .page-title {
            font-size: 2.7rem;
        }
    }

    </style>
        """
    )
)


# =========================================================
# PRESENTATION HELPERS
# =========================================================

def get_score_meta(score):
    if score >= 80:
        return "Excellent match", "#247A43", "Strong alignment with the target role."
    if score >= 60:
        return "Good match", "#2457E6", "Good alignment with a few areas to strengthen."
    if score >= 40:
        return "Moderate match", "#A96516", "Partial alignment with several gaps to address."
    return "Low match", "#A3312B", "Significant resume updates may be needed."


def safe_text(value):
    return html.escape(str(value))


def render_header():
    st.html(
        textwrap.dedent(
            """
        <div class="topbar">
            <div class="brand">Resume Match</div>
            <div class="top-links">
                <span>Analyze</span>
                <span>Resume Workbench</span>
            </div>
        </div>

        <div class="eyebrow">Resume analysis workbench</div>

        <div class="page-title">Does your resume<br>fit the role?</div>

        <div class="page-description">
            Compare a resume against a job description, find missing skills,
            inspect the evidence used by the RAG pipeline, and improve resume bullets.
        </div>
            """
        )
    )


def render_input_workspace():
    st.html(
        textwrap.dedent(
            """
        <div class="workspace-grid">
            <div class="workspace-cell">
                <div class="cell-label">01 · Resume</div>
                <div class="cell-note">
                    Upload the PDF you want to evaluate against the target role.
                </div>
            </div>
            <div class="workspace-cell">
                <div class="cell-label">02 · Job description</div>
                <div class="cell-note">
                    Paste the complete job description for the most useful comparison.
                </div>
            </div>
        </div>
            """
        )
    )

    col1, col2 = st.columns(2)

    with col1:
        resume_file = st.file_uploader(
            "Upload Resume",
            type=["pdf"],
            help="Upload your resume in PDF format.",
            label_visibility="collapsed"
        )

        if resume_file:
            st.html(
                f'<div class="file-name">✓ {safe_text(resume_file.name)}</div>'
            )

    with col2:
        job_description = st.text_area(
            "Job Description",
            height=170,
            placeholder="Paste the job description here...",
            label_visibility="collapsed"
        )

        word_count = len(job_description.split()) if job_description else 0
        st.html(
            f'<div class="word-count">{word_count} words</div>'
        )

    with st.container(key="cta_analyze"):
        analyze_col1, analyze_col2 = st.columns([4, 1])

        with analyze_col2:
            analyze_clicked = st.button(
                "Analyze →",
                type="primary",
                use_container_width=True
            )

    return resume_file, job_description, analyze_clicked


def render_empty_state():
    st.html(
        '<div class="empty-state">'
    )

    st.html(
        '<div class="empty-title">Your analysis will appear here.</div>'
    )

    st.html(
        '<div class="empty-copy">Upload a resume and add a job description to start the comparison.</div>'
    )

    # Render each step separately with Streamlit columns.
    # This prevents Streamlit Markdown from treating the HTML
    # as a code block.

    step_cols = st.columns(3)

    step_data = [
        ("01", "Upload resume"),
        ("02", "Retrieve & compare"),
        ("03", "Improve"),
    ]

    for col, (number, label) in zip(step_cols, step_data):
        with col:
            st.html(
                f'''
                <div class="step">
                    <div class="step-no">{number}</div>
                    <div class="step-text">{safe_text(label)}</div>
                </div>
                '''
            )

    st.html(
        '</div>'
    )


def render_score_section(
    ats_score,
    skill_score,
    semantic_score,
    experience_score,
    structure_score
):
    label, color, description = get_score_meta(ats_score)

    st.html(
        textwrap.dedent(
            f"""
        <div class="analysis-rule"></div>

        <div class="analysis-head">
            <div class="analysis-title">Analysis</div>
            <div class="analysis-status">● Analysis complete</div>
        </div>

        <div class="score-layout">
            <div>
                <div>
                    <span class="score-number">{ats_score}</span>
                    <span class="score-denom">/100</span>
                </div>
                <div class="score-label" style="color:{color};">
                    {safe_text(label)}
                </div>
            </div>

            <div>
                <div class="score-copy">
                    {safe_text(description)}
                </div>

                <div class="score-track">
                    <div
                        class="score-fill"
                        style="width:{max(0, min(100, ats_score))}%; background:{color};"
                    ></div>
                </div>
            </div>
        </div>
            """
        )
    )

    metrics = [
        ("Skills", skill_score, 100, f"{skill_score}%"),
        ("Semantic", semantic_score, 100, f"{semantic_score}%"),
        ("Experience", experience_score, 100, f"{experience_score}%"),
        ("Structure", structure_score, 10, f"{structure_score}/10"),
    ]

    metric_html = ""

    for metric_label, value, maximum, display in metrics:
        percentage = 0 if maximum == 0 else (value / maximum) * 100
        _, bar_color, _ = get_score_meta(percentage)

        metric_html += f"""
        <div class="metric">
            <div class="metric-label">{safe_text(metric_label)}</div>
            <div class="metric-value">{safe_text(display)}</div>
            <div class="metric-bar">
                <div
                    class="metric-bar-fill"
                    style="width:{max(0, min(100, percentage))}%; background:{bar_color};"
                ></div>
            </div>
        </div>
        """

    st.html(
        f'<div class="metric-grid">{metric_html}</div>'
    )


def render_section_nav():
    sections = [
        "Overview",
        "Skills",
        "ATS Insights",
        "RAG Evidence",
        "Bullet Optimizer"
    ]

    with st.container(key="section_nav"):
        cols = st.columns(len(sections))

        for col, section in zip(cols, sections):
            with col:
                active = st.session_state.active_section == section

                if st.button(
                    section,
                    key=f"nav_{section}",
                    type="primary" if active else "secondary",
                    use_container_width=True
                ):
                    st.session_state.active_section = section
                    st.rerun()


def render_skill_list(items, empty_message="None found."):
    if not items:
        return (
            f'<div class="muted-empty">{safe_text(empty_message)}</div>'
        )

    return "".join(
        f'<div class="skill-line">{safe_text(item)}</div>'
        for item in items
    )


def render_indicator(text, kind):
    st.html(
        f'<div class="indicator {kind}">{safe_text(text)}</div>'
    )


def render_section_heading(kicker, title, description):
    st.html(
        textwrap.dedent(
            f"""
        <div class="section-kicker">{safe_text(kicker)}</div>
        <div class="section-title">{safe_text(title)}</div>
        <div class="section-description">{safe_text(description)}</div>
            """
        )
    )


def render_footer():
    st.html(
        textwrap.dedent(
            """
        <div class="footer">
            RESUME MATCH · RAG · FAISS · LANGCHAIN · HUGGING FACE
        </div>
            """
        )
    )


# =========================================================
# HEADER
# =========================================================

render_header()


# =========================================================
# INPUT WORKSPACE
# =========================================================

resume_file, job_description, analyze_button = render_input_workspace()


# =========================================================
# ANALYSIS PIPELINE
# =========================================================

if analyze_button:

    if resume_file is None:
        st.error("Please upload your resume PDF.")

    elif not job_description.strip():
        st.error("Please enter a job description.")

    else:
        try:
            with st.spinner("Analyzing resume..."):

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

                if not resume_text.strip():
                    raise ValueError(
                        "No readable text was found in the uploaded PDF."
                    )

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

                required_skills = extract_job_skills(
                    job_description
                )

                # ---------------------------------------------
                # SKILL MATCH
                # ---------------------------------------------

                skill_result = calculate_skill_match(
                    required_skills=required_skills,
                    resume_text=resume_text
                )

                skill_score = skill_result["score"]

                # ---------------------------------------------
                # SEMANTIC MATCH
                # ---------------------------------------------

                embeddings = create_embeddings()

                chunk_scores = []

                for result in results:
                    score = calculate_semantic_similarity(
                        embeddings=embeddings,
                        job_description=job_description,
                        resume_text=result.page_content
                    )

                    chunk_scores.append(score)

                semantic_score = (
                    max(chunk_scores)
                    if chunk_scores
                    else 0
                )

                # ---------------------------------------------
                # OTHER SCORES
                # ---------------------------------------------

                experience_score = analysis.experience_match

                structure_score = calculate_structure_score(
                    resume_text
                )

                # ---------------------------------------------
                # FINAL ATS SCORE
                # ---------------------------------------------

                ats_score = calculate_ats_score(
                    skill_score=skill_score,
                    semantic_score=semantic_score,
                    experience_score=experience_score,
                    structure_score=structure_score
                )

            # ---------------------------------------------
            # SAVE RESULTS
            # ---------------------------------------------

            st.session_state.analysis_data = {
                "ats_score": ats_score,
                "skill_score": skill_score,
                "semantic_score": semantic_score,
                "experience_score": experience_score,
                "structure_score": structure_score,
                "skill_result": skill_result,
                "required_skills": required_skills,
                "analysis": analysis,
                "rag_sections": [
                    result.page_content
                    for result in results
                ],
                "job_description": job_description,
            }

            st.session_state.optimized_bullet = None
            st.session_state.optimized_original = None
            st.session_state.active_section = "Overview"

            st.rerun()

        except Exception as e:
            st.error(
                f"Analysis failed: {e}"
            )


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

    # -----------------------------------------------------
    # SCORE
    # -----------------------------------------------------

    render_score_section(
        ats_score=ats_score,
        skill_score=skill_score,
        semantic_score=semantic_score,
        experience_score=experience_score,
        structure_score=structure_score
    )

    # -----------------------------------------------------
    # NAV
    # -----------------------------------------------------

    render_section_nav()

    active = st.session_state.active_section

    # -----------------------------------------------------
    # OVERVIEW
    # -----------------------------------------------------

    if active == "Overview":

        matched_count = len(
            skill_result["matched_skills"]
        )

        total_count = len(required_skills)

        render_section_heading(
            "01 · Overview",
            "Match summary",
            "A compact view of where the resume fits and where attention is needed."
        )

        st.html(
            textwrap.dedent(
                f"""
            <div class="content-grid">

                <div class="content-cell">
                    <div class="content-label">Match result</div>
                    <div class="plain-copy">
                        Your resume matched
                        <strong>{matched_count}</strong>
                        of
                        <strong>{total_count}</strong>
                        skills identified in the job description.
                        The overall ATS compatibility score is
                        <strong>{ats_score}/100</strong>.
                    </div>
                </div>

                <div class="content-cell">
                    <div class="content-label">Experience & education</div>
                    <div class="plain-copy">
                        Experience match:
                        <strong>{experience_score}%</strong><br>
                        Education match:
                        <strong>{analysis.education_match}%</strong>
                    </div>
                </div>

            </div>
                """
            )
        )

        st.html("<br>")

        col1, col2 = st.columns(2)

        with col1:
            st.html(
                '<div class="content-label">Strong matches</div>'
            )

            st.html(
                render_skill_list(
                    skill_result["matched_skills"][:8],
                    "No strong matches identified yet."
                )
            )

        with col2:
            st.html(
                '<div class="content-label">Areas to improve</div>'
            )

            attention_items = (
                skill_result["missing_skills"][:5]
                + analysis.ats_issues[:2]
            )

            st.html(
                render_skill_list(
                    attention_items,
                    "No major gaps identified."
                )
            )

    # -----------------------------------------------------
    # SKILLS
    # -----------------------------------------------------

    elif active == "Skills":

        render_section_heading(
            "02 · Skills",
            "Role requirements",
            "Technical skills extracted from the job description and checked against the resume."
        )

        required_html = render_skill_list(
            required_skills,
            "No technical skills were extracted."
        )

        matched_html = render_skill_list(
            skill_result["matched_skills"],
            "No matching skills found."
        )

        skills_grid = f"""
        <div class="content-grid">
            <div class="content-cell">
                <div class="content-label">Required skills</div>
                {required_html}
            </div>

            <div class="content-cell">
                <div class="content-label">Matched skills</div>
                {matched_html}
            </div>
        </div>
        """

        st.html(skills_grid)

        st.markdown(
            '<div class="content-label" style="margin-top:25px;">Missing skills</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            render_skill_list(
                skill_result["missing_skills"],
                "No major missing skills found."
            ),
            unsafe_allow_html=True
        )

    # -----------------------------------------------------
    # ATS INSIGHTS
    # -----------------------------------------------------

    elif active == "ATS Insights":

        render_section_heading(
            "03 · ATS",
            "ATS insights",
            "Potential issues and practical changes that can improve alignment."
        )

        st.html(
            '<div class="content-label">ATS issues</div>'
        )

        if analysis.ats_issues:
            for issue in analysis.ats_issues:
                render_indicator(
                    issue,
                    "issue"
                )
        else:
            st.html(
                '<div class="muted-empty">No major ATS issues detected.</div>'
            )

        st.html(
            '<div class="content-label" style="margin-top:25px;">Recommended changes</div>'
        )

        if analysis.recommendations:
            for recommendation in analysis.recommendations:
                render_indicator(
                    recommendation,
                    "recommendation"
                )
        else:
            st.html(
                '<div class="muted-empty">No recommendations.</div>'
            )

        st.html(
            '<div class="priority-label">High priority</div>'
        )

        high_priority = [
            f"Consider adding or strengthening {skill} if you have genuine experience with it."
            for skill in skill_result["missing_skills"][:3]
        ]

        if high_priority:
            for item in high_priority:
                render_indicator(item, "high")
        else:
            st.html(
                '<div class="muted-empty">No high-priority gaps identified.</div>'
            )

        st.html(
            '<div class="priority-label">Medium priority</div>'
        )

        medium_priority = analysis.recommendations[:3]

        if medium_priority:
            for item in medium_priority:
                render_indicator(item, "medium")
        else:
            st.html(
                '<div class="muted-empty">No medium-priority improvements identified.</div>'
            )

        st.html(
            '<div class="priority-label">Low priority</div>'
        )

        low_priority = []

        if structure_score < 8:
            low_priority.append(
                "Improve resume structure by keeping important sections such as Skills, Projects, Education, and Experience clearly organized."
            )

        if semantic_score < 60:
            low_priority.append(
                "Improve keyword and content alignment between the resume and the target job description."
            )

        if low_priority:
            for item in low_priority:
                render_indicator(item, "low")
        else:
            st.html(
                '<div class="muted-empty">No major low-priority improvements identified.</div>'
            )

    # -----------------------------------------------------
    # RAG EVIDENCE
    # -----------------------------------------------------

    elif active == "RAG Evidence":

        render_section_heading(
            "04 · Retrieval",
            "RAG evidence",
            "The resume chunks retrieved through FAISS semantic search and passed to the analysis model."
        )

        if not rag_sections:
            st.html(
                '<div class="muted-empty">No retrieved resume context is available.</div>'
            )

        else:
            for i, section in enumerate(rag_sections, start=1):

                # IMPORTANT:
                # Render the actual text inside a dedicated HTML block
                # instead of st.write() inside an expander. This avoids
                # the visibility problem caused by inherited Streamlit CSS.

                escaped_section = safe_text(section)

                st.html(
                    textwrap.dedent(
                        f"""
                    <div class="rag-card">
                        <div class="rag-card-head">
                            <div class="rag-card-title">
                                Resume context {i:02d}
                            </div>
                            <div class="rag-card-meta">
                                FAISS retrieval
                            </div>
                        </div>

                        <div class="rag-text">{escaped_section}</div>
                    </div>
                        """
                    )
                )

    # -----------------------------------------------------
    # BULLET OPTIMIZER
    # -----------------------------------------------------

    elif active == "Bullet Optimizer":

        render_section_heading(
            "05 · Writing",
            "Rewrite a resume bullet",
            "Strengthen one bullet while preserving the original meaning and avoiding invented claims."
        )

        st.html(
            '<div class="bullet-label">Original bullet</div>'
        )

        bullet = st.text_area(
            "Original bullet",
            placeholder="Example: Developed a machine learning model using Python.",
            key="bullet_input",
            label_visibility="collapsed"
        )

        with st.container(key="cta_bullet"):

            button_col1, button_col2 = st.columns([4, 1])

            with button_col2:
                optimize_clicked = st.button(
                    "Improve bullet",
                    type="primary",
                    use_container_width=True
                )

        if optimize_clicked:

            if not bullet.strip():

                st.warning(
                    "Please enter a resume bullet."
                )

            else:

                try:

                    with st.spinner("Optimizing resume bullet..."):

                        optimized_bullet = optimize_bullet(
                            resume_bullet=bullet,
                            job_description=job_description_for_bullet
                        )

                    st.session_state.optimized_bullet = optimized_bullet
                    st.session_state.optimized_original = bullet

                    st.rerun()

                except Exception as e:

                    st.error(
                        f"Bullet optimization failed: {e}"
                    )

        if st.session_state.optimized_bullet:

            original = st.session_state.optimized_original
            optimized = st.session_state.optimized_bullet

            st.html(
                textwrap.dedent(
                    f"""
                <div class="before-after">

                    <div class="bullet-result">
                        <div class="bullet-result-label">Before</div>
                        <div class="bullet-result-text">
                            {safe_text(original)}
                        </div>
                    </div>

                    <div class="bullet-result">
                        <div class="bullet-result-label">After</div>
                        <div class="bullet-result-text">
                            {safe_text(optimized)}
                        </div>
                    </div>

                </div>
                    """
                )
            )


# =========================================================
# FOOTER
# =========================================================

render_footer()
