import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

FAISS_INDEX_PATH = "faiss_index"

PROMPT_TEMPLATE = """
You are a helpful assistant. Answer ONLY using the document context below.
If the answer isn't in the context, say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:
"""

def build_qa_chain():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Modern LCEL chain — replaces the deprecated RetrievalQA
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return {"chain": chain, "retriever": retriever}


def answer_question(qa_chain: dict, question: str) -> dict:
    chain = qa_chain["chain"]
    retriever = qa_chain["retriever"]

    # Run both in parallel: get answer and source docs
    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)

    sources = []
    seen = set()
    for doc in source_docs:
        page = doc.metadata.get("page", "?")
        source = doc.metadata.get("source", "document")
        key = (source, page)
        if key not in seen:
            seen.add(key)
            sources.append({
                "page": page,
                "source": source,
                "snippet": doc.page_content[:200].replace("\n", " "),
            })

    return {"answer": answer, "sources": sources}