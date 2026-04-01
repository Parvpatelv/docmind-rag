import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

FAISS_INDEX_PATH = "faiss_index"

def get_smart_splitter(doc_text: str):
    """
    Choose chunk size based on document length and density.
    Short docs (resumes, 1-2 pages): smaller chunks = more precise retrieval.
    Long docs (reports, books): larger chunks = more context per chunk.
    """
    total_chars = len(doc_text)

    if total_chars < 5000:
        # Short doc — very small chunks, high overlap
        return RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=80,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
        )
    elif total_chars < 30000:
        # Medium doc — balanced
        return RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
        )
    else:
        # Long doc — larger chunks for more context
        return RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=200,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
        )


def ingest_pdf(pdf_path: str) -> int:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from pathlib import Path
    import os

    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    # Get full text to decide chunk size
    full_text = " ".join(doc.page_content for doc in documents)

    # Clean up common PDF extraction noise
    import re
    for doc in documents:
        doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()
        doc.page_content = re.sub(r'(\w)-\n(\w)', r'\1\2', doc.page_content)  # fix hyphenation
        # Add source filename to metadata for citation display
        doc.metadata["source"] = os.path.basename(pdf_path)

    # Filter out nearly empty pages (headers/footers only)
    documents = [d for d in documents if len(d.page_content.strip()) > 50]

    splitter = get_smart_splitter(full_text)
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (strategy: {len(full_text)} chars)")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    if Path(FAISS_INDEX_PATH).exists():
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"Saved index → ./{FAISS_INDEX_PATH}/")
    return len(chunks)