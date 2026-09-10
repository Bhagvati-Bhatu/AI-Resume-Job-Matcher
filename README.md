# AI Resume–Job Matcher

An AI-powered Resume–Job Matching and ATS Optimization System that analyzes a resume against a job description using Retrieval-Augmented Generation (RAG), semantic search, embeddings, FAISS, and Generative AI.

## Overview

The system helps candidates understand how well their resume matches a target job description.

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
- Resume improvement plan
- RAG retrieval evidence
- AI-powered resume bullet optimization

## Features

### 1. Resume PDF Upload

Upload a resume in PDF format.

The system extracts the text from the PDF using PyMuPDF.

### 2. RAG-Based Resume Retrieval

The extracted resume is divided into smaller chunks.

The chunks are converted into vector embeddings using:

`sentence-transformers/all-MiniLM-L6-v2`

The embeddings are stored in a FAISS vector database.

When a job description is provided, semantic similarity search retrieves the most relevant resume sections.

### 3. AI Resume Analysis

The retrieved resume sections and job description are passed to GPT-OSS-20B.

The model evaluates:

- Resume-job compatibility
- Matched skills
- Missing skills
- Experience alignment
- Education alignment
- ATS issues
- Resume improvement recommendations

### 4. AI Job Skill Extraction

The system uses Generative AI to extract important technical skills, tools, frameworks, platforms, and methodologies from the job description.

These skills are then compared against the resume.

### 5. ATS Scoring

The system calculates an overall ATS score using:

- Skill Match
- Semantic Match
- Experience Match
- Resume Structure

The final score is displayed from 0–100.

### 6. Resume Improvement Plan

The system categorizes improvements into:

- High Priority
- Medium Priority
- Low Priority

This helps candidates focus on the most important changes first.

### 7. RAG Evidence

The application displays the resume sections retrieved by the RAG pipeline.

This makes the retrieval process transparent and demonstrates which parts of the resume were used for the AI analysis.

### 8. AI Resume Bullet Optimizer

Users can enter an existing resume bullet and receive an improved ATS-friendly version.

The optimizer is instructed not to invent:

- Technologies
- Metrics
- Achievements
- Experience

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
              Sentence Transformers
                        |
                        v
                  FAISS Vector DB
                        |
                        |
Job Description --------+
                        |
                        v
                 Similarity Search
                        |
                        v
              Relevant Resume Chunks
                        |
                        v
                   GPT-OSS-20B
                        |
            +-----------+-----------+
            |           |           |
            v           v           v
       Resume       Skill Gap    ATS Analysis
       Analysis     Analysis
            |           |           |
            +-----------+-----------+
                        |
                        v
                 ATS Score & Report
                        |
            +-----------+-----------+
            |                       |
            v                       v
    Improvement Plan        Bullet Optimizer
```

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Web application interface |
| LangChain | LLM and RAG pipeline |
| Hugging Face | Embeddings and LLM |
| Sentence Transformers | Text embeddings |
| FAISS | Vector similarity search |
| PyMuPDF | PDF text extraction |
| Pydantic | Structured data validation |
| NumPy | Numerical calculations |
| GPT-OSS-20B | Generative AI analysis |

## Project Structure

```text
AI-Resume-Job-Matcher/
│
├── app.py
├── requirements.txt
├── .gitignore
├── .env
│
├── .streamlit/
│   └── config.toml
│
├── data/
│   ├── resumes/
│   └── job_description.txt
│
├── src/
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── llm.py
│   ├── rag_pipeline.py
│   ├── ats_scorer.py
│   └── bullet_optimizer.py
│
└── venv/
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project

```bash
cd AI-Resume-Job-Matcher
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure Hugging Face API Token

Create a `.env` file in the project root:

```text
HF_TOKEN=your_huggingface_token
```

Never commit the `.env` file to GitHub.

It is already included in `.gitignore`.

### 7. Run the application

```bash
streamlit run app.py
```

The application will open in your browser.

## How the RAG Pipeline Works

The application follows these steps:

1. Extract text from the uploaded resume.
2. Split the resume into smaller chunks.
3. Generate embeddings for each chunk.
4. Store the embeddings in FAISS.
5. Convert the job description into an embedding.
6. Perform similarity search.
7. Retrieve the most relevant resume chunks.
8. Pass the retrieved chunks to the LLM.
9. Generate structured resume analysis.
10. Calculate ATS and skill matching scores.

This allows the LLM to focus on the most relevant parts of the resume instead of blindly processing unrelated content.

## ATS Score

The final ATS score combines multiple components:

```text
Final ATS Score =
    Skill Match × 40%
  + Semantic Match × 30%
  + Experience Match × 20%
  + Resume Structure × 10%
```

The resume structure score is normalized from a 10-point scale to a 100-point scale before calculating the final score.

## Example Workflow

```text
Upload Resume
      ↓
Paste Job Description
      ↓
Click "Analyze Resume"
      ↓
AI extracts job skills
      ↓
RAG retrieves relevant resume sections
      ↓
LLM analyzes compatibility
      ↓
ATS score generated
      ↓
Skill gaps identified
      ↓
Improvement plan generated
      ↓
Resume bullets can be optimized
```

## Important Design Principle

The system is designed to avoid encouraging candidates to add false information to their resumes.

AI-generated recommendations should only be applied when they accurately reflect the candidate's real skills, experience, and achievements.

## Future Improvements

Potential future improvements include:

- Multi-resume comparison
- Job description URL support
- Resume section-level scoring
- Keyword frequency analysis
- Downloadable ATS reports
- Resume template generation
- Authentication
- Deployment with persistent storage
- Improved semantic skill matching
- Dashboard with historical resume scores

## Disclaimer

ATS scores generated by this application are estimates intended to help improve resume-job alignment.

Different Applicant Tracking Systems may use different scoring and ranking methods.

## Author

Bhagvati

Built using Python, LangChain, Hugging Face, FAISS, RAG, and Streamlit.