import re

import numpy as np

from src.embeddings import create_embeddings


# ---------------------------------------------------------
# TEXT NORMALIZATION
# ---------------------------------------------------------

def normalize_text(text):
    """
    Normalize text so small formatting differences
    do not prevent skill matching.
    """

    text = text.lower()

    # Replace & with "and"
    text = text.replace("&", " and ")

    # Remove brackets
    text = re.sub(
        r"[\(\)\[\]\{\}]",
        " ",
        text
    )

    # Replace hyphens, underscores and slashes
    # with spaces
    text = re.sub(
        r"[-_/]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ---------------------------------------------------------
# COSINE SIMILARITY
# ---------------------------------------------------------

def cosine_similarity(vector1, vector2):
    """
    Calculate cosine similarity between two vectors.
    """

    vector1 = np.array(vector1)

    vector2 = np.array(vector2)

    denominator = (
        np.linalg.norm(vector1)
        * np.linalg.norm(vector2)
    )

    if denominator == 0:
        return 0

    similarity = (
        np.dot(
            vector1,
            vector2
        )
        / denominator
    )

    return similarity


# ---------------------------------------------------------
# SKILL MATCHING
# ---------------------------------------------------------

def skill_matches_resume(
    skill,
    resume_text,
    embeddings,
    threshold=0.65
):
    """
    Check whether a required skill is present
    in the resume.

    Uses:

    1. Exact normalized matching
    2. Semantic embedding similarity
    """

    normalized_resume = normalize_text(
        resume_text
    )

    normalized_skill = normalize_text(
        skill
    )

    # -----------------------------------------------------
    # STEP 1: EXACT MATCHING
    # -----------------------------------------------------

    pattern = (
        r"\b"
        + re.escape(normalized_skill)
        + r"\b"
    )

    if re.search(
        pattern,
        normalized_resume
    ):
        return True

    # -----------------------------------------------------
    # STEP 2: SEMANTIC MATCHING
    # -----------------------------------------------------

    skill_embedding = embeddings.embed_query(
        skill
    )

    resume_embedding = embeddings.embed_query(
        resume_text
    )

    similarity = cosine_similarity(
        skill_embedding,
        resume_embedding
    )

    return similarity >= threshold


# ---------------------------------------------------------
# SKILL MATCHING SCORE
# ---------------------------------------------------------

def calculate_skill_match(
    required_skills,
    resume_text,
    embeddings=None
):
    """
    Calculate the percentage of required job skills
    found in the resume.

    Embeddings are optional so existing function calls
    continue to work.
    """

    # Create embedding model if it was not supplied
    if embeddings is None:
        embeddings = create_embeddings()

    matched_skills = []

    missing_skills = []

    # Check every required skill
    for skill in required_skills:

        if skill_matches_resume(
            skill,
            resume_text,
            embeddings
        ):

            matched_skills.append(skill)

        else:

            missing_skills.append(skill)

    # -----------------------------------------------------
    # CALCULATE SCORE
    # -----------------------------------------------------

    if not required_skills:

        score = 0

    else:

        score = (
            len(matched_skills)
            / len(required_skills)
        ) * 100

    return {
        "score": round(score),

        "matched_skills": matched_skills,

        "missing_skills": missing_skills
    }


# ---------------------------------------------------------
# RESUME STRUCTURE SCORE
# ---------------------------------------------------------

def calculate_structure_score(resume_text):
    """
    Calculate a basic resume structure score.

    Checks for:

    - Education
    - Skills
    - Projects
    - Experience
    - Certifications
    - Email
    - Phone number
    """

    score = 0

    sections = [
        "education",
        "skills",
        "projects",
        "experience",
        "certifications"
    ]

    text = resume_text.lower()

    # -----------------------------------------------------
    # SECTION CHECK
    # -----------------------------------------------------

    for section in sections:

        if section in text:

            score += 1

    # -----------------------------------------------------
    # EMAIL CHECK
    # -----------------------------------------------------

    if re.search(
        r"\b[\w\.-]+@[\w\.-]+\.\w+\b",
        resume_text
    ):

        score += 1

    # -----------------------------------------------------
    # PHONE CHECK
    # -----------------------------------------------------

    if re.search(
        r"\b\d{10}\b",
        resume_text
    ):

        score += 1

    # Maximum score = 10
    return min(score, 10)


# ---------------------------------------------------------
# SEMANTIC SIMILARITY
# ---------------------------------------------------------

def calculate_semantic_similarity(
    embeddings,
    job_description,
    resume_text
):
    """
    Calculate semantic similarity between the
    job description and the resume.
    """

    # -----------------------------------------------------
    # JOB EMBEDDING
    # -----------------------------------------------------

    job_embedding = embeddings.embed_query(
        job_description
    )

    # -----------------------------------------------------
    # RESUME EMBEDDING
    # -----------------------------------------------------

    resume_embedding = embeddings.embed_query(
        resume_text
    )

    # -----------------------------------------------------
    # COSINE SIMILARITY
    # -----------------------------------------------------

    similarity = cosine_similarity(
        job_embedding,
        resume_embedding
    )

    # Convert similarity to 0-100 score
    score = (
        similarity * 100
    )

    # Keep score between 0 and 100
    score = max(
        0,
        min(
            100,
            score
        )
    )

    return round(score)


# ---------------------------------------------------------
# FINAL ATS SCORE
# ---------------------------------------------------------

def calculate_ats_score(
    skill_score,
    semantic_score,
    experience_score,
    structure_score
):
    """
    Calculate final ATS score using weighted components.

    Skill matching       = 40%
    Semantic similarity  = 30%
    Experience match     = 20%
    Resume structure     = 10%
    """

    # Convert structure score from 0-10
    # to 0-100
    structure_score_100 = (
        structure_score * 10
    )

    # -----------------------------------------------------
    # WEIGHTED SCORE
    # -----------------------------------------------------

    final_score = (

        skill_score * 0.40

        + semantic_score * 0.30

        + experience_score * 0.20

        + structure_score_100 * 0.10
    )

    return round(final_score)