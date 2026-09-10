from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.llm import HuggingFaceLLM


def optimize_bullet(resume_bullet, job_description):

    llm = HuggingFaceLLM()

    prompt = PromptTemplate(
        template="""
You are an expert ATS resume writer.

Improve the resume bullet so it is stronger, more professional,
and better aligned with the job description.

JOB DESCRIPTION:

{job_description}

ORIGINAL RESUME BULLET:

{resume_bullet}

FOLLOW THESE RULES:

1. Keep the original meaning.
2. Do NOT invent technologies.
3. Do NOT invent metrics or percentages.
4. Do NOT invent achievements.
5. Do NOT claim experience that is not in the original bullet.
6. Use strong action verbs.
7. Use relevant keywords from the job description only if supported.
8. Make the bullet concise and ATS-friendly.
9. Prefer this structure:

Action + What you did + Technology/Method + Purpose/Result

10. Return ONLY the improved bullet.
11. Do NOT provide explanations.
12. Do NOT use quotation marks.

IMPROVED BULLET:
""",
        input_variables=[
            "job_description",
            "resume_bullet"
        ]
    )

    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({
        "job_description": job_description,
        "resume_bullet": resume_bullet
    })

    return result.strip()