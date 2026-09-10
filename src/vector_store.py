from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.embeddings import create_embeddings


def create_vector_store(resume_text):

    document = Document(
        page_content=resume_text
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(
        [document]
    )

    embeddings = create_embeddings()

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store