import re
import numpy as np


def calculate_skill_match(
    required_skills,
    resume_text
):

    resume_text = resume_text.lower()

    matched_skills = []
    missing_skills = []

    for skill in required_skills:

        skill_lower = skill.lower()

        if skill_lower in resume_text:

            matched_skills.append(skill)

        else:

            missing_skills.append(skill)

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

def calculate_structure_score(resume_text):

    score = 0

    sections = [
        "education",
        "skills",
        "projects",
        "experience",
        "certifications"
    ]

    text = resume_text.lower()

    for section in sections:

        if section in text:
            score += 1

    # Email
    if re.search(
        r"\b[\w.-]+@[\w.-]+\.\w+\b",
        resume_text
    ):
        score += 1

    # Phone number
    if re.search(
        r"\b\d{10}\b",
        resume_text
    ):
        score += 1

    return min(score, 10)


def calculate_semantic_similarity(
    embeddings,
    job_description,
    resume_text
):

    job_embedding = embeddings.embed_query(
        job_description
    )

    resume_embedding = embeddings.embed_query(
        resume_text
    )

    job_vector = np.array(
        job_embedding
    )

    resume_vector = np.array(
        resume_embedding
    )

    cosine_similarity = np.dot(
        job_vector,
        resume_vector
    ) / (
        np.linalg.norm(job_vector)
        * np.linalg.norm(resume_vector)
    )

    score = max(
        0,
        min(
            100,
            cosine_similarity * 100
        )
    )

    return round(score)

def calculate_ats_score(
    skill_score,
    semantic_score,
    experience_score,
    structure_score
):
    structure_score_100 = structure_score * 10

    final_score = (
        skill_score * 0.40
        + semantic_score * 0.30
        + experience_score * 0.20
        + structure_score_100 * 0.10
    )

    return round(final_score)