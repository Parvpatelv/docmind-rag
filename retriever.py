import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
FAISS_INDEX_PATH = "faiss_index"

# History-aware prompt — includes previous conversation turns
PROMPT_TEMPLATE = """
You are a helpful assistant. Answer ONLY using the document context below.
If the answer isn't in the context, say "I don't have enough information in the document."

Previous conversation:
{chat_history}

Document context:
{context}

Current question: {question}

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
    retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={
        "k": 4,
        "score_threshold": 0.3,  # only return chunks that are at least 30% similar
    }
    )
    return {"retriever": retriever, "llm": llm, "vectorstore": vectorstore}


def format_history(history: list) -> str:
    """Convert history list to a readable string for the prompt."""
    if not history:
        return "No previous conversation."
    lines = []
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)


def answer_question(qa_chain: dict, question: str, history: list = []) -> dict:
    retriever = qa_chain["retriever"]
    llm = qa_chain["llm"]
    vectorstore = qa_chain["vectorstore"]  # add this to build_qa_chain return

    # Get docs WITH scores (0.0 = identical, higher = less similar for L2)
    docs_with_scores = vectorstore.similarity_search_with_score(question, k=4)
    source_docs = [doc for doc, score in docs_with_scores]
    context = "\n\n".join(doc.page_content for doc in source_docs)

    prompt_text = PROMPT_TEMPLATE.format(
        chat_history=format_history(history),
        context=context,
        question=question,
    )
    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt_text)])
    answer = response.content

    # Build rich source objects with relevance score
    sources = []
    seen = set()
    for doc, score in docs_with_scores:
        page = doc.metadata.get("page", "?")
        source_file = doc.metadata.get("source", "document")
        key = (source_file, page)
        if key not in seen:
            seen.add(key)
            # Convert L2 distance to a 0-100 relevance score
            relevance = max(0, round((1 - score / 2) * 100))
            sources.append({
                "page": page,
                "source": source_file,
                "snippet": doc.page_content[:300].replace("\n", " "),
                "relevance": relevance,
            })

    # Sort by relevance descending
    sources.sort(key=lambda x: x["relevance"], reverse=True)
    return {"answer": answer, "sources": sources}