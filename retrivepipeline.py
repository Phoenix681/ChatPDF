import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from indexingpipe import get_active_collection, CHROMA_PATH

load_dotenv()

# --- Global model clients (cheap to init, no network call at startup) ---
llm = ChatGoogleGenerativeAI(temperature=0.2, model="gemini-3.6-flash")
embedding_models = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def get_vector_db() -> Chroma:
    """
    Lazily connects to whichever collection is currently active.
    Raises a clear error if no PDF has been indexed yet.
    """
    collection = get_active_collection()
    if not collection:
        raise RuntimeError(
            "No document indexed yet. "
            "Please upload a PDF first via the /upload-pdf endpoint."
        )
    try:
        return Chroma(
            collection_name=collection,
            embedding_function=embedding_models,
            persist_directory=str(CHROMA_PATH)
        )
    except Exception:
        raise RuntimeError(
            f"Collection '{collection}' not found in Chroma. "
            "It may have been deleted. Please re-upload your PDF."
        )


def ask_tutor(user_query: str) -> dict:
    """Retrieves context and answers queries. Returns answer + cited page numbers."""
    vector_db = get_vector_db()
    search_results = vector_db.similarity_search(query=user_query, k=4)

    # Collect unique page numbers from retrieved chunks
    pages = sorted(set(
        str(r.metadata.get("page_label") or r.metadata.get("page", ""))
        for r in search_results
        if r.metadata.get("page_label") or r.metadata.get("page") is not None
    ))

    context = "\n\n".join(
        [
            f"Page Content: {result.page_content}\n"
            f"Page Number: {result.metadata.get('page_label', 'Unknown')}\n"
            f"File Location: {result.metadata.get('source', 'Unknown')}"
            for result in search_results
        ]
    )

    SYSTEM_PROMPT = f"""
You are an expert technical tutor. Answer the query using ONLY the provided context.
Do NOT include page numbers inside your answer text — they will be shown separately.
If the answer cannot be determined from the context, state that the information
is unavailable in the current index.

Context:
{context}
"""

    response = llm.invoke([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ])

    # response.content can be a plain string or a list of content blocks
    # (e.g. [{"type": "text", "text": "..."}]) depending on the model/provider.
    # Normalize it to a plain string before returning it to the frontend.
    raw_content = response.content
    if isinstance(raw_content, list):
        answer_text = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw_content
        )
    else:
        answer_text = str(raw_content)

    return {"answer": answer_text, "pages": pages}