# AI Resume–Job Matcher

An AI-powered Resume–Job Matching and ATS Optimization System that analyzes a resume against a target job description using RAG, semantic search, embeddings, FAISS, LangChain, and Generative AI.

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

- OOP ↔ Object-Oriented Programming
- STLC ↔ Software Testing Life Cycle
- SDLC ↔ Software Development Life Cycle
- ChatGPT ↔ Chat GPT
- Hugging Face ↔ HuggingFace
- Power BI ↔ PowerBI

This reduces false skill mismatches caused by formatting or naming differences.

### 6. ATS Scoring

The final ATS score combines:

| Component | Weight |
|---|---:|
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
       +--------+--------+
       |        |        |
       v        v        v
    Resume   Skill Gap   ATS
    Analysis  Analysis  Analysis
       |        |        |
       +--------+--------+
                |
                v
          ATS Score & Report
                |
        +-------+-------+
        |               |
        v               v
 Improvement Plan   Bullet Optimizer