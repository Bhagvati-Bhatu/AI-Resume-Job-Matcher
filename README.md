# AI Resume–Job Matcher

An AI-powered Resume–Job Matching and ATS Optimization System that analyzes a resume against a target job description using Retrieval-Augmented Generation (RAG), semantic search, embeddings, FAISS, LangChain, and Generative AI.

## Live Demo

[Launch the AI Resume–Job Matcher](https://ai-resume-job-matcher-kr5j2sohxkd6qgyayehur8.streamlit.app/)

## Overview

The AI Resume–Job Matcher helps candidates understand how well their resume aligns with a target job description.

The application analyzes the resume and provides:

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

## Key Features

### Resume PDF Analysis

Upload a resume PDF and extract its text using PyMuPDF.

### RAG-Based Resume Retrieval

The extracted resume is split into smaller chunks, converted into vector embeddings, stored in FAISS, and retrieved using semantic similarity against the target job description.

### AI Resume Analysis

GPT-OSS-20B analyzes the retrieved resume context and job description to evaluate:

- Resume-job compatibility
- Matched skills
- Missing skills
- Experience alignment
- Education alignment
- ATS issues
- Resume improvement recommendations

### Job Skill Extraction

The system uses an LLM to extract relevant technical skills, tools, technologies, frameworks, platforms, and methodologies from the job description.

### Intelligent Skill Matching

The ATS engine uses normalized and alias-aware skill matching.

Examples:

- `OOP` ↔ `Object-Oriented Programming`
- `STLC` ↔ `Software Testing Life Cycle`
- `SDLC` ↔ `Software Development Life Cycle`
- `ChatGPT` ↔ `Chat GPT`
- `Hugging Face` ↔ `HuggingFace`
- `Power BI` ↔ `PowerBI`

### ATS Scoring

The final ATS score combines:

| Component | Weight |
| :--- | :---: |
| Skill Match | 40% |
| Semantic Match | 30% |
| Experience Match | 20% |
| Resume Structure | 10% |

### RAG Evidence

The application displays the resume sections retrieved by the RAG pipeline, providing transparency into which resume content was used during the analysis.

### AI Resume Bullet Optimizer

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
```

## Technology Stack

| Technology | Purpose |
| :--- | :--- |
| Python | Core programming language |
| Streamlit | Web application interface |
| LangChain | LLM and RAG pipeline |
| LangChain Community | FAISS integration |
| LangChain Hugging Face | Hugging Face integration |
| LangChain Text Splitters | Resume text chunking |
| Hugging Face | Embeddings and LLM integration |
| Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| FAISS | Vector similarity search |
| PyMuPDF | PDF text extraction |
| Pydantic | Structured output validation |
| NumPy | Numerical calculations |
| GPT-OSS-20B | Generative AI analysis |
| Hugging Face Inference Providers | LLM inference |
| python-dotenv | Environment variable management |
| Git | Version control |
| GitHub | Source code hosting |
| Streamlit Community Cloud | Deployment |

## Project Structure

```text
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
```

## File Responsibilities

| File | Responsibility |
| :--- | :--- |
| `app.py` | Streamlit application and user interface |
| `document_loader.py` | PDF and text extraction |
| `embeddings.py` | Hugging Face embedding model |
| `vector_store.py` | Resume chunking and FAISS vector database |
| `llm.py` | GPT-OSS-20B integration |
| `rag_pipeline.py` | RAG analysis and structured LLM output |
| `ats_scorer.py` | ATS scoring and skill matching |
| `bullet_optimizer.py` | AI resume bullet optimization |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Files excluded from Git |
| `.streamlit/config.toml` | Streamlit configuration |

## RAG Pipeline

1. Extract text from the uploaded resume.
2. Split the resume into smaller chunks.
3. Generate embeddings for each chunk.
4. Store the embeddings in FAISS.
5. Convert the job description into an embedding.
6. Perform similarity search.
7. Retrieve the most relevant resume chunks.
8. Pass the retrieved context to the LLM.
9. Generate structured resume analysis.
10. Calculate ATS and skill-matching scores.
11. Generate recommendations and improvement suggestions.

## How RAG Is Used

Instead of sending the entire resume directly to the LLM, the system first retrieves the most relevant resume sections based on the target job description.

```text
Resume
   |
   v
Extract Text
   |
   v
Split into Chunks
   |
   v
Generate Embeddings
   |
   v
FAISS Vector Store
   |
   v
Job Description Query
   |
   v
Similarity Search
   |
   v
Relevant Resume Chunks
   |
   v
LLM
   |
   v
Resume Analysis
```

This makes the project a practical RAG application rather than a simple LLM chatbot.

## Embedding Model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The model converts resume and job-description text into numerical vector representations for semantic similarity search.

## Vector Database

FAISS is used as the vector database.

Resume chunks are converted into embeddings and stored in FAISS. The system then performs similarity search to retrieve the most relevant resume sections for a given job description.

## ATS Score Calculation

```text
Final ATS Score =
    Skill Match × 40%
  + Semantic Match × 30%
  + Experience Match × 20%
  + Resume Structure × 10%
```

The final score is displayed on a 0–100 scale.

## ATS Components

### Skill Match — 40%

Measures how many required job skills are represented in the resume.

### Semantic Match — 30%

Measures semantic similarity between the resume and target job description.

### Experience Match — 20%

Evaluates how well the candidate's experience aligns with the role.

### Resume Structure — 10%

Evaluates important resume sections and overall resume organization.

## Skill Matching

The system performs normalized and alias-aware matching.

For example:

```text
Resume:
Object-Oriented Programming

Job Description:
OOP
```

These can be recognized as the same skill.

Another example:

```text
Resume:
Hugging Face

Job Description:
HuggingFace
```

These variations can also be recognized as a match.

## Structured LLM Output

The resume analysis uses Pydantic to validate the LLM response.

The application also includes retry handling when the LLM returns malformed JSON.

```text
LLM Response
     |
     v
JSON Parsing
     |
     +---- Valid ----> Pydantic Validation
     |
     +---- Invalid
             |
             v
       Strict JSON Retry
             |
             v
       Pydantic Validation
```

## AI Resume Bullet Optimizer

The bullet optimizer takes:

```text
Original Resume Bullet
+
Target Job Description
```

and generates a stronger ATS-friendly version.

The optimizer preserves the original meaning and avoids intentionally adding unsupported:

- Technologies
- Metrics
- Achievements
- Experience

## Application Workflow

```text
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
```

## User Interface

The application provides several analysis sections.

### Overview

Displays the overall resume-job compatibility and key analysis results.

### Skills

Shows:

- Required skills
- Matched skills
- Missing skills
- Skill match percentage

### ATS Insights

Displays ATS-related issues and recommendations for improving the resume.

### RAG Evidence

Shows the resume sections retrieved by semantic search.

### Bullet Optimizer

Allows users to improve individual resume bullets using the target job description.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Bhagvati-Bhatu/AI-Resume-Job-Matcher.git
```

### 2. Open the Project

```bash
cd AI-Resume-Job-Matcher
```

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

### 4. Activate the Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### macOS/Linux

```bash
source venv/bin/activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

## Hugging Face Configuration

Create a Hugging Face API token.

Create a `.env` file in the project root:

```text
HF_TOKEN=your_huggingface_token
```

Never commit your `.env` file to GitHub.

The `.env` file is included in `.gitignore`.

## Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

## Usage

### Step 1 — Upload Resume

Upload your resume as a PDF file.

### Step 2 — Enter Job Description

Paste the complete job description into the job description field.

### Step 3 — Analyze

Click the `ANALYZE` button.

The system will:

- Extract resume text
- Create resume chunks
- Generate embeddings
- Build a FAISS vector store
- Extract job skills
- Retrieve relevant resume sections
- Analyze the resume using GPT-OSS-20B
- Calculate the ATS score
- Identify skill gaps
- Generate recommendations

### Step 4 — Review Results

Review:

- Overview
- Skills
- ATS Insights
- RAG Evidence
- Bullet Optimizer

### Step 5 — Optimize Resume Bullets

Enter an existing resume bullet and generate an improved ATS-friendly version.

## Example Input

### Job Description

```text
Looking for a candidate with Python, SQL, machine learning,
data analysis, Git, and problem-solving skills.
```

### Resume

```text
Python developer with experience in machine learning projects,
Git, data analysis, and software development.
```

## Example Output

```text
ATS Score: 78 / 100

Skill Match: 80%
Semantic Match: 76%
Experience Match: 75%
Resume Structure: 90%
```

Example skill analysis:

```text
Matched Skills:
- Python
- Machine Learning
- Git

Missing Skills:
- SQL
- Data Analysis
```

The actual results depend on the uploaded resume and job description.

## Technologies Demonstrated

This project demonstrates practical implementation of:

- Python
- Generative AI
- Retrieval-Augmented Generation
- Large Language Models
- Prompt Engineering
- Semantic Search
- Vector Databases
- Text Embeddings
- LangChain
- Hugging Face
- FAISS
- Streamlit
- Pydantic
- PDF Processing
- ATS Scoring
- Information Retrieval

## Challenges Solved

### Malformed LLM Responses

LLMs can occasionally return invalid JSON.

The application handles this using:

- JSON extraction
- Pydantic validation
- Strict retry prompts
- Error handling

### Skill Name Variations

The same skill can appear under different names.

Normalized and alias-aware matching reduces false mismatches.

### Relevant Resume Retrieval

A resume may contain information unrelated to a specific job.

The RAG pipeline retrieves relevant resume sections before sending context to the LLM.

### Combining AI With Traditional ATS Logic

The project combines:

```text
Traditional ATS Scoring
+
Semantic Similarity
+
RAG
+
Generative AI
```

This provides a more explainable resume analysis workflow.

## Design Principles

The system is designed to help candidates improve resume-job alignment without encouraging fabricated information.

AI-generated recommendations should only be applied when they accurately reflect the candidate's real skills, experience, and achievements.

## Security

Sensitive credentials should never be stored directly in source code.

The project uses:

```text
.env
```

for local Hugging Face authentication.

The following files are excluded from Git:

```text
.env
venv/
data/resumes/resume.pdf
```

This helps prevent personal information and API credentials from being uploaded to the public repository.

## Deployment

The application is deployed using Streamlit Community Cloud.

### Live Application

[AI Resume–Job Matcher](https://ai-resume-job-matcher-kr5j2sohxkd6ggyavehur8.streamlit.app/)

For deployment, the Hugging Face token should be configured securely through the platform's secrets settings rather than committed to the repository.

## GitHub Repository

[View Source Code on GitHub](https://github.com/Bhagvati-Bhatu/AI-Resume-Job-Matcher)

## Future Improvements

- Multi-resume comparison
- Job description URL support
- Resume section-level scoring
- Keyword frequency analysis
- Downloadable ATS reports
- Resume template generation
- Authentication
- Historical resume score tracking
- Improved semantic skill matching
- Support for multiple resume formats
- Job recommendations based on resume similarity
- Resume version comparison
- PDF report generation
- More advanced semantic skill extraction

## Limitations

The ATS score generated by this application is an estimate.

Different Applicant Tracking Systems may use different:

- Keyword matching strategies
- Ranking algorithms
- Resume parsing methods
- Scoring systems

Therefore, the generated score should be treated as an optimization indicator rather than an exact representation of a company's ATS score.

The quality of the AI analysis also depends on the quality and completeness of the resume and job description.

## Disclaimer

This application is designed to assist candidates with resume optimization and job matching.

AI-generated suggestions should always be reviewed by the candidate before being added to a resume.

Candidates should only include skills, technologies, achievements, and experience that accurately represent their real background.

## License

This project is intended for educational and portfolio purposes.

## Author

**Bhagvati**

Built using Python, Streamlit, LangChain, Hugging Face, FAISS, RAG, and GPT-OSS-20B.

## Acknowledgements

This project uses open-source tools and technologies from the Python, LangChain, Hugging Face, FAISS, Sentence Transformers, PyMuPDF, Pydantic, and Streamlit ecosystems.
