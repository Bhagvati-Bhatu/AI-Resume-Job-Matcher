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

    # Remove markdown code fences
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

    # Find JSON object
    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL
    )

    if not match:
        raise ValueError(
            "No JSON object found in LLM response."
        )

    json_text = match.group(0)

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


    # Create chain
    chain = prompt | llm | StrOutputParser()


    # Call LLM
    raw_result = chain.invoke({

        "job_description": job_description,

        "resume_sections": resume_sections

    })


    try:

        # Extract JSON
        data = extract_json(
            raw_result
        )


        # Validate with Pydantic
        result = ResumeAnalysis(
            **data
        )


        return result


    except Exception as e:

        print(
            "\n========== RAW LLM RESPONSE ==========\n"
        )

        print(raw_result)

        raise ValueError(
            f"Could not parse LLM response as JSON: {e}"
        )

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

    raw_result = chain.invoke({
        "job_description": job_description
    })

    try:

        data = extract_json(raw_result)

        return data["skills"]

    except Exception as e:

        print(
            "\n========== RAW SKILL RESPONSE ==========\n"
        )

        print(raw_result)

        raise ValueError(
            f"Could not extract job skills: {e}"
        )