import json
import re

from pydantic import BaseModel, Field

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.llm import HuggingFaceLLM


class ResumeAnalysis(BaseModel):

    match_score: int = Field(
        description="Overall resume and job match score from 0 to 100"
    )

    matched_skills: list[str] = Field(
        description="Skills from the job description present in the resume"
    )

    missing_skills: list[str] = Field(
        description="Important job skills missing from the resume"
    )

    experience_match: int = Field(
        description="Experience match score from 0 to 100"
    )

    education_match: int = Field(
        description="Education match score from 0 to 100"
    )

    ats_issues: list[str] = Field(
        description="Potential ATS issues"
    )

    recommendations: list[str] = Field(
        description="Actionable resume improvement recommendations"
    )


def extract_json(text):

    if not text or not text.strip():
        raise ValueError(
            "LLM returned an empty response."
        )

    text = re.sub(
        r"```json\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```\s*",
        "",
        text
    )

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if not match:
        raise ValueError(
            "No JSON object found in LLM response."
        )

    json_text = match.group(0).strip()

    return json.loads(json_text)


def analyze_resume(
    job_description,
    resume_sections
):

    llm = HuggingFaceLLM()

    prompt = PromptTemplate(
        template="""
You are an expert ATS resume evaluator.

Compare the resume with the job description.

JOB DESCRIPTION:

{job_description}

RELEVANT RESUME SECTIONS:

{resume_sections}

Analyze the candidate using ONLY the information provided.

IMPORTANT RULES:

- Do not invent skills.
- Do not invent experience.
- Do not invent certifications.
- Do not invent achievements.
- Missing information should be treated as missing.
- Scores must be integers from 0 to 100.
- Use at most 5 matched skills.
- Use at most 5 missing skills.
- Use at most 3 ATS issues.
- Use at most 5 recommendations.
- Keep recommendations short.
- Return ONLY valid JSON.
- Do NOT use markdown.
- Do NOT write explanations before or after the JSON.
- Make sure every JSON key and string is enclosed in double quotes.
- Make sure all arrays and objects are properly closed.
- Make sure commas are placed correctly.
- Do not include trailing commas.

Use EXACTLY this structure:

{{
    "match_score": 0,
    "matched_skills": [],
    "missing_skills": [],
    "experience_match": 0,
    "education_match": 0,
    "ats_issues": [],
    "recommendations": []
}}
""",
        input_variables=[
            "job_description",
            "resume_sections"
        ]
    )

    chain = prompt | llm | StrOutputParser()

    input_data = {
        "job_description": job_description,
        "resume_sections": resume_sections
    }

    # ---------------------------------------------------------
    # FIRST ATTEMPT
    # ---------------------------------------------------------

    raw_result = chain.invoke(input_data)

    try:

        data = extract_json(raw_result)

        result = ResumeAnalysis(**data)

        return result

    except Exception as first_error:

        print(
            "\n========== FIRST ANALYSIS RESPONSE ==========\n"
        )

        print(raw_result)

        print(
            "\n========== FIRST PARSING ERROR ==========\n"
        )

        print(first_error)

    # ---------------------------------------------------------
    # SECOND ATTEMPT
    # ---------------------------------------------------------

    retry_prompt = PromptTemplate(
        template="""
You are an expert ATS resume evaluator.

Your previous response contained invalid JSON.

Create the resume analysis again.

JOB DESCRIPTION:

{job_description}

RELEVANT RESUME SECTIONS:

{resume_sections}

STRICT OUTPUT RULES:

1. Return ONLY one valid JSON object.
2. Do NOT use markdown.
3. Do NOT use ```json.
4. Do NOT add explanations.
5. Do NOT add text before or after the JSON.
6. Every key must use double quotes.
7. Every string must use double quotes.
8. Arrays must be valid JSON arrays.
9. Objects must be properly closed.
10. Every item must be separated by a comma.
11. Do not use trailing commas.
12. Scores must be integers between 0 and 100.
13. Do not invent information.
14. Use at most 5 matched skills.
15. Use at most 5 missing skills.
16. Use at most 3 ATS issues.
17. Use at most 5 recommendations.

Return EXACTLY:

{{
    "match_score": 0,
    "matched_skills": [],
    "missing_skills": [],
    "experience_match": 0,
    "education_match": 0,
    "ats_issues": [],
    "recommendations": []
}}
""",
        input_variables=[
            "job_description",
            "resume_sections"
        ]
    )

    retry_chain = retry_prompt | llm | StrOutputParser()

    retry_result = retry_chain.invoke(input_data)

    try:

        data = extract_json(retry_result)

        result = ResumeAnalysis(**data)

        return result

    except Exception as second_error:

        print(
            "\n========== RETRY ANALYSIS RESPONSE ==========\n"
        )

        print(retry_result)

        print(
            "\n========== RETRY PARSING ERROR ==========\n"
        )

        print(second_error)

        raise ValueError(
            "The AI returned invalid analysis data after two attempts. "
            "Please click ANALYZE again."
        )


def fallback_extract_job_skills(job_description):

    common_skills = [
        "Python",
        "Java",
        "C++",
        "C",
        "JavaScript",
        "TypeScript",
        "SQL",
        "R",
        "Machine Learning",
        "ML",
        "Deep Learning",
        "Artificial Intelligence",
        "AI",
        "Generative AI",
        "GenAI",
        "NLP",
        "Computer Vision",
        "Data Science",
        "Data Analytics",
        "LLM",
        "RAG",
        "LangChain",
        "LlamaIndex",
        "FAISS",
        "Hugging Face",
        "OpenAI",
        "Django",
        "Flask",
        "FastAPI",
        "React",
        "Angular",
        "Node.js",
        "MySQL",
        "PostgreSQL",
        "MongoDB",
        "Redis",
        "SQL Server",
        "AWS",
        "Azure",
        "GCP",
        "Docker",
        "Kubernetes",
        "Git",
        "GitHub",
        "CI/CD",
        "Data Structures",
        "Algorithms",
        "Object Oriented Programming",
        "OOP",
        "REST APIs",
        "APIs",
        "Microservices",
        "System Design",
        "Software Architecture",
        "Testing",
        "Unit Testing",
        "PyTest",
        "Pandas",
        "NumPy",
        "Matplotlib",
        "Power BI",
        "Tableau",
        "Spark",
        "PySpark"
    ]

    job_text = job_description.lower()

    found_skills = []

    for skill in common_skills:

        if skill.lower() in job_text:

            if skill not in found_skills:
                found_skills.append(skill)

    return found_skills[:20]


def extract_job_skills(job_description):

    llm = HuggingFaceLLM()

    prompt = PromptTemplate(
        template="""
You are an expert technical recruiter.

Extract the important technical skills, tools, technologies,
frameworks, platforms, and methodologies explicitly required
or strongly preferred in the job description.

JOB DESCRIPTION:

{job_description}

RULES:

- Return ONLY valid JSON.
- Do not invent skills.
- Extract only skills actually mentioned in the job description.
- Remove duplicates.
- Use standard skill names.
- Return at most 20 skills.
- Every JSON key and string must use double quotes.
- Make sure the JSON is properly closed.
- Do not use markdown.

Use exactly this format:

{{
    "skills": []
}}
""",
        input_variables=[
            "job_description"
        ]
    )

    chain = prompt | llm | StrOutputParser()

    input_data = {
        "job_description": job_description
    }

    # ---------------------------------------------------------
    # FIRST ATTEMPT
    # ---------------------------------------------------------

    raw_result = chain.invoke(input_data)

    try:

        data = extract_json(raw_result)

        skills = data.get("skills", [])

        if not isinstance(skills, list):
            raise ValueError(
                "The skills field is not a list."
            )

        return skills[:20]

    except Exception as first_error:

        print(
            "\n========== FIRST SKILL RESPONSE ==========\n"
        )

        print(raw_result)

        print(
            "\n========== FIRST SKILL PARSING ERROR ==========\n"
        )

        print(first_error)

    # ---------------------------------------------------------
    # SECOND ATTEMPT
    # ---------------------------------------------------------

    retry_prompt = PromptTemplate(
        template="""
Extract technical skills from the job description.

JOB DESCRIPTION:

{job_description}

Return ONLY valid JSON.

Do not write explanations.
Do not use markdown.
Do not use ```json.
Do not invent skills.
Extract only skills explicitly mentioned.
Remove duplicates.
Return at most 20 skills.

Use EXACTLY:

{{
    "skills": []
}}
""",
        input_variables=[
            "job_description"
        ]
    )

    retry_chain = retry_prompt | llm | StrOutputParser()

    retry_result = retry_chain.invoke(input_data)

    try:

        data = extract_json(retry_result)

        skills = data.get("skills", [])

        if not isinstance(skills, list):
            raise ValueError(
                "The skills field is not a list."
            )

        return skills[:20]

    except Exception as second_error:

        print(
            "\n========== RETRY SKILL RESPONSE ==========\n"
        )

        print(retry_result)

        print(
            "\n========== RETRY SKILL PARSING ERROR ==========\n"
        )

        print(second_error)

        # -----------------------------------------------------
        # FALLBACK KEYWORD EXTRACTION
        # -----------------------------------------------------

        print(
            "\n========== USING FALLBACK SKILL EXTRACTION ==========\n"
        )

        return fallback_extract_job_skills(
            job_description
        )