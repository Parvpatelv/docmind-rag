import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

FAISS_INDEX_PATH = "faiss_index"

def ingest_pdf(pdf_path: str) -> int:
    # Step 1: Load all pages from the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    # Step 2: Split pages into overlapping chunks
    # chunk_size=1000 chars, chunk_overlap=200 preserves context at boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Step 3: Create vector embeddings for every chunk
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Step 4: Build or update the FAISS index and save it to disk
    if Path(FAISS_INDEX_PATH).exists():
        # Add to existing index so previous PDFs aren't lost
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"Saved index to ./{FAISS_INDEX_PATH}/")
    return len(chunks)
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

FAISS_INDEX_PATH = "faiss_index"

def ingest_pdf(pdf_path: str) -> int:
    # Step 1: Load all pages from the PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    # Step 2: Split pages into overlapping chunks
    # chunk_size=1000 chars, chunk_overlap=200 preserves context at boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Step 3: Create vector embeddings for every chunk
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # Step 4: Build or update the FAISS index and save it to disk
    if Path(FAISS_INDEX_PATH).exists():
        # Add to existing index so previous PDFs aren't lost
        vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        vectorstore.add_documents(chunks)
    else:
        vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"Saved index to ./{FAISS_INDEX_PATH}/")
    return len(chunks)