import re
import numpy as np


# ---------------------------------------------------------
# SKILL NORMALIZATION
# ---------------------------------------------------------

SKILL_ALIASES = {
    "object oriented programming": [
        "oop",
        "object-oriented programming",
        "object oriented programming",
        "object-oriented programming (oop)"
    ],

    "software development life cycle": [
        "sdlc",
        "software development life cycle",
        "software development lifecycle"
    ],

    "software testing life cycle": [
        "stlc",
        "software testing life cycle",
        "software testing lifecycle"
    ],

    "test case writing": [
        "test case writing",
        "test case design",
        "test cases"
    ],

    "github copilot": [
        "github copilot",
        "github-copilot",
        "github copilot"
    ],

    "chatgpt": [
        "chatgpt",
        "chat gpt"
    ],

    "machine learning": [
        "machine learning",
        "ml"
    ],

    "deep learning": [
        "deep learning",
        "dl"
    ],

    "artificial intelligence": [
        "artificial intelligence",
        "ai"
    ],

    "generative ai": [
        "generative ai",
        "genai",
        "generative artificial intelligence"
    ],

    "natural language processing": [
        "natural language processing",
        "nlp"
    ],

    "computer vision": [
        "computer vision",
        "cv"
    ],

    "data analytics": [
        "data analytics",
        "data analysis"
    ],

    "data science": [
        "data science"
    ],

    "large language model": [
        "large language model",
        "large language models",
        "llm",
        "llms"
    ],

    "rest api": [
        "rest api",
        "rest apis",
        "restful api",
        "restful apis"
    ],

    "unit testing": [
        "unit testing",
        "unit tests"
    ],

    "python": [
        "python"
    ],

    "java": [
        "java"
    ],

    "javascript": [
        "javascript",
        "js"
    ],

    "typescript": [
        "typescript",
        "ts"
    ],

    "sql": [
        "sql"
    ],

    "mysql": [
        "mysql"
    ],

    "postgresql": [
        "postgresql",
        "postgres"
    ],

    "mongodb": [
        "mongodb",
        "mongo db"
    ],

    "pandas": [
        "pandas"
    ],

    "numpy": [
        "numpy"
    ],

    "matplotlib": [
        "matplotlib"
    ],

    "power bi": [
        "power bi",
        "powerbi"
    ],

    "tableau": [
        "tableau"
    ],

    "langchain": [
        "langchain"
    ],

    "llamaindex": [
        "llamaindex",
        "llama index"
    ],

    "hugging face": [
        "hugging face",
        "huggingface"
    ],

    "faiss": [
        "faiss"
    ],

    "docker": [
        "docker"
    ],

    "kubernetes": [
        "kubernetes",
        "k8s"
    ],

    "amazon web services": [
        "amazon web services",
        "aws"
    ],

    "microsoft azure": [
        "microsoft azure",
        "azure"
    ],

    "google cloud platform": [
        "google cloud platform",
        "gcp",
        "google cloud"
    ],

    "git": [
        "git"
    ],

    "github": [
        "github",
        "git hub"
    ],

    "continuous integration": [
        "continuous integration",
        "ci"
    ],

    "continuous delivery": [
        "continuous delivery",
        "cd"
    ],

    "continuous integration and continuous delivery": [
        "ci/cd",
        "ci cd",
        "continuous integration and continuous delivery"
    ]
}


def normalize_text(text):
    """
    Normalize text so small formatting differences
    do not prevent skill matching.
    """

    text = text.lower()

    text = text.replace("&", " and ")

    text = re.sub(
        r"[\(\)\[\]\{\}]",
        " ",
        text
    )

    text = re.sub(
        r"[-_/]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def skill_matches_resume(skill, resume_text):
    """
    Check whether a required skill is present in the resume.

    Uses:
    1. Normalized exact matching
    2. Alias matching
    """

    normalized_resume = normalize_text(resume_text)
    normalized_skill = normalize_text(skill)

    # Direct normalized match
    pattern = r"\b" + re.escape(normalized_skill) + r"\b"

    if re.search(pattern, normalized_resume):
        return True

    # Alias matching
    for canonical_skill, aliases in SKILL_ALIASES.items():

        normalized_aliases = [
            normalize_text(alias)
            for alias in aliases
        ]

        if (
            normalized_skill == normalize_text(canonical_skill)
            or normalized_skill in normalized_aliases
        ):

            for alias in normalized_aliases:

                alias_pattern = (
                    r"\b"
                    + re.escape(alias)
                    + r"\b"
                )

                if re.search(
                    alias_pattern,
                    normalized_resume
                ):
                    return True

    return False


# ---------------------------------------------------------
# SKILL MATCHING SCORE
# ---------------------------------------------------------

def calculate_skill_match(
    required_skills,
    resume_text
):

    matched_skills = []
    missing_skills = []

    for skill in required_skills:

        if skill_matches_resume(
            skill,
            resume_text
        ):

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


# ---------------------------------------------------------
# RESUME STRUCTURE SCORE
# ---------------------------------------------------------

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

    # 10 digit phone number
    if re.search(
        r"\b\d{10}\b",
        resume_text
    ):
        score += 1

    return min(score, 10)


# ---------------------------------------------------------
# SEMANTIC SIMILARITY
# ---------------------------------------------------------

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

    denominator = (
        np.linalg.norm(job_vector)
        * np.linalg.norm(resume_vector)
    )

    if denominator == 0:

        return 0

    cosine_similarity = (
        np.dot(
            job_vector,
            resume_vector
        )
        / denominator
    )

    score = max(
        0,
        min(
            100,
            cosine_similarity * 100
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

    structure_score_100 = (
        structure_score * 10
    )

    final_score = (

        skill_score * 0.40

        + semantic_score * 0.30

        + experience_score * 0.20

        + structure_score_100 * 0.10
    )

    return round(final_score)