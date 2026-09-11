# AI Resume–Job Matcher

An AI-powered Resume–Job Matching and ATS Optimization System that analyzes a resume against a target job description using Retrieval-Augmented Generation (RAG), semantic search, embeddings, FAISS, LangChain, and Generative AI.

## Live Demo

[Launch the AI Resume–Job Matcher](https://ai-resume-job-matcher-kr5j2sohxkd6ggyavehur8.streamlit.app/)

## Overview

The application helps candidates understand how well their resume aligns with a target job description.

It analyzes the resume and provides:

- Overall ATS compatibility score
- Skill match score
- Semantic similarity score
- Experience match score
- Resume structure score
- Matched skills
- Missing skills
- ATS issues
- AI-generated recommendations
- RAG retrieval evidence
- AI-powered resume bullet optimization

## Screenshots

### Resume Analysis

![Resume Analysis](screenshots/analysis.png)

### Skills and Skill Gap Analysis

![Skills Analysis](screenshots/skills.png)

### ATS Insights

![ATS Insights](screenshots/ats-insights.png)

### RAG Evidence

![RAG Evidence](screenshots/rag-evidence.png)

### AI Resume Bullet Optimizer

![Bullet Optimizer](screenshots/bullet-optimizer.png)

## Why This Project?

Traditional resume screening systems often rely heavily on keyword matching.

This project combines keyword-based ATS scoring with semantic similarity and Retrieval-Augmented Generation to provide a more meaningful comparison between a resume and a job description.

The system helps candidates understand:

- Which required skills they already have
- Which skills are missing
- How closely their experience matches the role
- Potential ATS issues
- Which resume bullets can be improved

## Key Features

### 1. Resume PDF Analysis

Upload a resume PDF and extract its text using PyMuPDF.

### 2. RAG-Based Resume Retrieval

The extracted resume is:

1. Split into smaller chunks
2. Converted into vector embeddings
3. Stored in a FAISS vector database
4. Retrieved using semantic similarity against the target job description

This allows the LLM to focus on the most relevant sections of the resume.

### 3. AI Resume Analysis

Relevant resume sections and the job description are analyzed using GPT-OSS-20B.

The system evaluates:

- Resume-job compatibility
- Matched skills
- Missing skills
- Experience alignment
- Education alignment
- ATS issues
- Resume improvement recommendations

### 4. Job Skill Extraction

The system uses an LLM to extract important:

- Technical skills
- Tools
- Technologies
- Frameworks
- Platforms
- Methodologies

from the job description.

The extracted skills are then compared against the resume.

### 5. Intelligent Skill Matching

The ATS engine uses normalized skill matching and aliases to handle variations such as:

- `OOP` ↔ `Object-Oriented Programming`
- `STLC` ↔ `Software Testing Life Cycle`
- `SDLC` ↔ `Software Development Life Cycle`
- `ChatGPT` ↔ `Chat GPT`
- `Hugging Face` ↔ `HuggingFace`
- `Power BI` ↔ `PowerBI`

This reduces false skill mismatches caused by formatting or naming differences.

### 6. ATS Scoring

The final ATS score combines the following components:

| Component | Weight |
| :--- | :---: |
| Skill Match | 40% |
| Semantic Match | 30% |
| Experience Match | 20% |
| Resume Structure | 10% |

The final score is displayed on a 0–100 scale.

### 7. RAG Evidence

The application displays the resume sections retrieved by the RAG pipeline.

This provides transparency into which parts of the resume were used during the analysis.

### 8. AI Resume Bullet Optimizer

Users can enter an existing resume bullet and receive an improved ATS-friendly version.

The optimizer is instructed not to invent:

- Technologies
- Metrics
- Achievements
- Experience

The goal is to improve clarity and keyword alignment while preserving the original meaning.

## Key Highlights

- Built an end-to-end RAG pipeline for resume-job matching
- Implemented FAISS-based semantic retrieval over resume sections
- Used Hugging Face embeddings for semantic search
- Integrated GPT-OSS-20B for resume analysis and recommendations
- Implemented structured LLM output validation using Pydantic
- Developed normalized and alias-aware skill matching
- Built a weighted ATS scoring engine
- Added an AI-powered resume bullet optimizer
- Deployed the application using Streamlit Community Cloud

## System Architecture

```text
                    Resume PDF
                        |
                        v
                PDF Text Extraction
                        |
                        v
                   Text Chunking
                        |
                        v
              Hugging Face Embeddings
                        |
                        v
                  FAISS Vector DB
                        |
                        |
                Job Description
                        |
                        v
                 Semantic Search
                        |
                        v
              Relevant Resume Chunks
                        |
                        v
                  GPT-OSS-20B
                        |
          +-------------+-------------+
          |             |             |
          v             v             v
       Resume        Skill Gap       ATS
       Analysis       Analysis      Analysis
          |             |             |
          +-------------+-------------+
                        |
                        v
                 ATS Score & Report
                        |
               +--------+--------+
               |                 |
               v                 v
        Improvement Plan    Bullet Optimizer

Technology Stack
Technology	Purpose
Python	Core programming language
Streamlit	Web application interface
LangChain	LLM and RAG pipeline
Hugging Face	Embeddings and LLM
Sentence Transformers	Text embeddings
FAISS	Vector similarity search
PyMuPDF	PDF text extraction
Pydantic	Structured output validation
NumPy	Numerical calculations
GPT-OSS-20B	Generative AI analysis
Project Structure
AI-Resume-Job-Matcher/
│
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   └── job_description.txt
│
└── src/
    ├── document_loader.py
    ├── embeddings.py
    ├── vector_store.py
    ├── llm.py
    ├── rag_pipeline.py
    ├── ats_scorer.py
    └── bullet_optimizer.py
RAG Pipeline

The application follows these steps:

Extract text from the uploaded resume.
Split the resume into smaller chunks.
Generate embeddings for each chunk.
Store the embeddings in FAISS.
Convert the job description into an embedding.
Perform similarity search.
Retrieve the most relevant resume chunks.
Pass the retrieved context to the LLM.
Generate structured resume analysis.
Calculate ATS and skill-matching scores.
Generate recommendations and improvement suggestions.
ATS Score Calculation
Final ATS Score =
    Skill Match × 40%
  + Semantic Match × 30%
  + Experience Match × 20%
  + Resume Structure × 10%

The structure score is normalized from a 10-point scale to a 100-point scale before calculating the final score.

Example Workflow
Upload Resume
      |
      v
Paste Job Description
      |
      v
Click ANALYZE
      |
      v
Extract Job Skills
      |
      v
Create Resume Embeddings
      |
      v
FAISS Similarity Search
      |
      v
Retrieve Relevant Resume Sections
      |
      v
GPT-OSS-20B Analysis
      |
      v
Calculate ATS Score
      |
      v
Identify Skill Gaps
      |
      v
Generate Recommendations
      |
      v
Optimize Resume Bullets
Installation
1. Clone the Repository
git clone https://github.com/Bhagvati-Bhatu/AI-Resume-Job-Matcher.git
2. Open the Project
cd AI-Resume-Job-Matcher
3. Create a Virtual Environment
python -m venv venv
4. Activate the Virtual Environment

Windows:

venv\Scripts\activate

macOS/Linux:

source venv/bin/activate
5. Install Dependencies
pip install -r requirements.txt
6. Configure Hugging Face

Create a .env file in the project root:

HF_TOKEN=your_huggingface_token

Never commit the .env file to GitHub.

The .env file is already included in .gitignore.

7. Run the Application
streamlit run app.py

The application will open in your browser.

Design Principles

The system is designed to help candidates improve resume-job alignment without encouraging fabricated information.

AI-generated recommendations should only be applied when they accurately reflect the candidate's real skills, experience, and achievements.

Future Improvements

Potential future improvements include:

Multi-resume comparison
Job description URL support
Resume section-level scoring
Keyword frequency analysis
Downloadable ATS reports
Resume template generation
Authentication
Historical resume score tracking
Improved semantic skill matching
Disclaimer

ATS scores generated by this application are estimates intended to help improve resume-job alignment.

Different Applicant Tracking Systems may use different scoring and ranking methods.

Author

Bhagvati

Built using Python, Streamlit, LangChain, Hugging Face, FAISS, RAG, and GPT-OSS-20B.
