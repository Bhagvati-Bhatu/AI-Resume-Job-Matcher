import streamlit as st

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
    page_title="AI Resume Matcher",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title("AI Resume–Job Matcher")

st.markdown("### ATS Optimization System")

st.write(
    "Analyze your resume against a job description using "
    "RAG, semantic search, embeddings, and Generative AI."
)

st.divider()


# =========================================================
# INPUT SECTION
# =========================================================

st.subheader("Resume & Job Description")

col1, col2 = st.columns(2)


with col1:

    resume_file = st.file_uploader(
        "Upload Resume",
        type=["pdf"],
        help="Upload your resume in PDF format."
    )

    if resume_file:

        st.success(
            f"Resume uploaded: {resume_file.name}"
        )


with col2:

    job_description = st.text_area(
        "Job Description",
        height=200,
        placeholder=(
            "Paste the complete job description here..."
        )
    )


st.write("")


# =========================================================
# ANALYZE BUTTON
# =========================================================

analyze_button = st.button(
    "Analyze Resume",
    type="primary",
    use_container_width=True
)


# =========================================================
# ANALYSIS
# =========================================================

if analyze_button:

    if resume_file is None:

        st.error(
            "Please upload your resume PDF."
        )

    elif not job_description.strip():

        st.error(
            "Please enter a job description."
        )

    else:

        with st.spinner(
            "Analyzing resume using RAG and Generative AI..."
        ):

            # ---------------------------------------------
            # SAVE RESUME
            # ---------------------------------------------

            resume_path = "temp_resume.pdf"

            with open(
                resume_path,
                "wb"
            ) as file:

                file.write(
                    resume_file.getbuffer()
                )


            # ---------------------------------------------
            # EXTRACT RESUME TEXT
            # ---------------------------------------------

            resume_text = extract_text_from_pdf(
                resume_path
            )


            # ---------------------------------------------
            # CREATE VECTOR STORE
            # ---------------------------------------------

            vector_store = create_vector_store(
                resume_text
            )


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


            if chunk_scores:

                semantic_score = max(
                    chunk_scores
                )

            else:

                semantic_score = 0


            # ---------------------------------------------
            # OTHER SCORES
            # ---------------------------------------------

            experience_score = (
                analysis.experience_match
            )

            structure_score = (
                calculate_structure_score(
                    resume_text
                )
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


        # =================================================
        # ANALYSIS COMPLETED
        # =================================================

        st.success(
            "Resume analysis completed successfully."
        )

        st.divider()


        # =================================================
        # OVERALL ATS SCORE
        # =================================================

        st.subheader(
            "Overall ATS Score"
        )

        score_col1, score_col2, score_col3 = st.columns(
            [1, 2, 1]
        )

        with score_col2:

            st.metric(
                "ATS Compatibility",
                f"{ats_score}/100"
            )

            st.progress(
                ats_score / 100
            )


            if ats_score >= 80:

                st.success(
                    "Excellent match — your resume is highly aligned with the job."
                )

            elif ats_score >= 60:

                st.info(
                    "Good match — a few improvements could strengthen your resume."
                )

            elif ats_score >= 40:

                st.warning(
                    "Moderate match — consider improving your skills and experience alignment."
                )

            else:

                st.error(
                    "Low match — significant resume improvements are recommended."
                )


        st.divider()


        # =================================================
        # SCORE BREAKDOWN
        # =================================================

        st.subheader(
            "Score Breakdown"
        )

        col1, col2, col3, col4 = st.columns(4)


        with col1:

            st.metric(
                "Skill Match",
                f"{skill_score}/100"
            )


        with col2:

            st.metric(
                "Semantic Match",
                f"{semantic_score}/100"
            )


        with col3:

            st.metric(
                "Experience Match",
                f"{experience_score}/100"
            )


        with col4:

            st.metric(
                "Resume Structure",
                f"{structure_score}/10"
            )


        # =================================================
        # ATS SCORE BREAKDOWN CHART
        # =================================================

        st.markdown(
            "#### ATS Score Visualization"
        )

        chart_data = {
            "Skill Match": skill_score,
            "Semantic Match": semantic_score,
            "Experience Match": experience_score,
            "Resume Structure": structure_score * 10
        }

        st.bar_chart(
            chart_data,
            horizontal=True
        )

        st.caption(
            "Resume Structure is normalized from 10 to 100 "
            "for visualization."
        )


        st.divider()


        # =================================================
        # JOB SKILLS EXTRACTED BY AI
        # =================================================

        st.subheader(
            "Job Skills Extracted by AI"
        )

        st.write(
            "Technical skills identified from the job "
            "description using Generative AI."
        )


        if required_skills:

            skill_text = " • ".join(
                required_skills
            )

            st.info(
                skill_text
            )

        else:

            st.warning(
                "No technical skills were extracted "
                "from the job description."
            )


        st.divider()


        # =================================================
        # SKILL ANALYSIS
        # =================================================

        st.subheader(
            "Skill Analysis"
        )

        skill_col1, skill_col2 = st.columns(2)


        # -------------------------------------------------
        # MATCHED SKILLS
        # -------------------------------------------------

        with skill_col1:

            st.markdown(
                "#### Matched Skills"
            )

            if skill_result["matched_skills"]:

                for skill in skill_result["matched_skills"]:

                    st.success(
                        f"✓ {skill}"
                    )

            else:

                st.info(
                    "No matching skills found."
                )


        # -------------------------------------------------
        # MISSING SKILLS
        # -------------------------------------------------

        with skill_col2:

            st.markdown(
                "#### Missing Skills"
            )

            if skill_result["missing_skills"]:

                for skill in skill_result["missing_skills"]:

                    st.error(
                        f"✗ {skill}"
                    )

            else:

                st.success(
                    "No major missing skills found."
                )


        st.divider()


        # =================================================
        # ATS ISSUES
        # =================================================

        st.subheader(
            "ATS Issues"
        )

        if analysis.ats_issues:

            for issue in analysis.ats_issues:

                st.warning(
                    issue
                )

        else:

            st.success(
                "No major ATS issues detected."
            )


        st.divider()


        # =================================================
        # AI RECOMMENDATIONS
        # =================================================

        st.subheader(
            "AI Recommendations"
        )

        if analysis.recommendations:

            for recommendation in analysis.recommendations:

                st.info(
                    recommendation
                )

        else:

            st.success(
                "No recommendations."
            )


        st.divider()


        # =================================================
        # RESUME IMPROVEMENT PLAN
        # =================================================

        st.subheader(
            "Resume Improvement Plan"
        )

        st.write(
            "Prioritized actions to improve your resume's "
            "alignment with the target job."
        )


        # -------------------------------------------------
        # HIGH PRIORITY
        # -------------------------------------------------

        st.markdown(
            "#### 🔴 High Priority"
        )

        high_priority = []

        for skill in skill_result["missing_skills"][:3]:

            high_priority.append(
                f"Consider adding or strengthening **{skill}** "
                "if you have genuine experience with it."
            )


        if high_priority:

            for item in high_priority:

                st.error(
                    item
                )

        else:

            st.success(
                "No high-priority skill gaps identified."
            )


        # -------------------------------------------------
        # MEDIUM PRIORITY
        # -------------------------------------------------

        st.markdown(
            "#### 🟠 Medium Priority"
        )

        medium_priority = []

        for recommendation in analysis.recommendations[:3]:

            medium_priority.append(
                recommendation
            )


        if medium_priority:

            for item in medium_priority:

                st.warning(
                    item
                )

        else:

            st.info(
                "No medium-priority improvements identified."
            )


        # -------------------------------------------------
        # LOW PRIORITY
        # -------------------------------------------------

        st.markdown(
            "#### 🟢 Low Priority"
        )

        low_priority = []


        if structure_score < 8:

            low_priority.append(
                "Improve resume structure by ensuring "
                "important sections such as Skills, Projects, "
                "Education, and Experience are clearly organized."
            )


        if semantic_score < 60:

            low_priority.append(
                "Improve keyword and content alignment between "
                "your resume and the target job description."
            )


        if low_priority:

            for item in low_priority:

                st.info(
                    item
                )

        else:

            st.success(
                "No major low-priority improvements identified."
            )


        st.divider()


        # =================================================
        # RAG EVIDENCE
        # =================================================

        st.subheader(
            "RAG Evidence"
        )

        st.write(
            "Resume sections retrieved by the RAG system "
            "for job-specific analysis."
        )


        for i, result in enumerate(
            results,
            start=1
        ):

            with st.expander(
                f"Retrieved Resume Section {i}"
            ):

                st.write(
                    result.page_content
                )


        st.divider()


        # =================================================
        # AI BULLET OPTIMIZER
        # =================================================

        st.subheader(
            "AI Resume Bullet Optimizer"
        )

        st.write(
            "Rewrite a resume bullet to make it more "
            "professional and ATS-friendly."
        )


        bullet = st.text_area(
            "Resume Bullet",
            placeholder=(
                "Example: Developed a machine learning "
                "model using Python."
            )
        )


        optimize_button = st.button(
            "Optimize Bullet"
        )


        if optimize_button:

            if not bullet.strip():

                st.warning(
                    "Please enter a resume bullet."
                )

            else:

                with st.spinner(
                    "Optimizing resume bullet..."
                ):

                    optimized_bullet = optimize_bullet(
                        resume_bullet=bullet,
                        job_description=job_description
                    )


                st.markdown(
                    "#### Original Bullet"
                )

                st.write(
                    bullet
                )


                st.markdown(
                    "#### AI-Optimized Bullet"
                )

                st.success(
                    optimized_bullet
                )


        st.divider()


        # =================================================
        # TECHNOLOGY
        # =================================================

        st.caption(
            "Built with LangChain • Hugging Face • "
            "FAISS • Sentence Transformers • RAG • "
            "GPT-OSS-20B"
        )