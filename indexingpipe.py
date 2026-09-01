import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import re
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import chromadb
import time
from dotenv import load_dotenv

load_dotenv()

# Local on-disk folder where Chroma persists its data (no server/Docker needed)
CHROMA_PATH = Path(__file__).parent / "chroma_db"
# Flat file that tracks whichever collection is currently active
ACTIVE_COLLECTION_FILE = Path(__file__).parent / ".active_collection"


def get_active_collection() -> str | None:
    """Returns the currently active collection name, or None if none exists."""
    if ACTIVE_COLLECTION_FILE.exists():
        return ACTIVE_COLLECTION_FILE.read_text().strip() or None
    return None


def set_active_collection(name: str):
    """Persists the active collection name to disk."""
    ACTIVE_COLLECTION_FILE.write_text(name)


def make_collection_name(filename: str) -> str:
    """
    Converts a PDF filename into a safe Qdrant collection name.
    e.g. 'Node.js Guide (2024).pdf' -> 'nodejs_guide_2024'
    """
    stem = Path(filename).stem                     # strip .pdf extension
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", stem)    # replace special chars with _
    slug = slug.strip("_").lower()[:60]            # trim, lowercase, max 60 chars
    return slug or "document"


def index_document(file_name: str = "Nodejs.pdf", display_name: str | None = None) -> str:
    """
    Loads a PDF, chunks it, and saves it to a NEW Chroma collection.
    Deletes the previous collection so only one PDF is active at a time.
    Returns the new collection name.
    """
    pdf_path = Path(file_name)
    if not pdf_path.is_absolute():
        pdf_path = Path.cwd() / file_name

    print(f"[INFO] Initializing indexing process for: {pdf_path}")

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    new_collection = make_collection_name(display_name or pdf_path.name)

    try:
        # 1. Delete the previous collection if one exists
        previous = get_active_collection()
        if previous:
            existing = [c.name for c in client.list_collections()]
            if previous in existing:
                print(f"[INFO] Deleting previous collection: '{previous}'")
                client.delete_collection(previous)

        # 2. Load the PDF
        loader = PyPDFLoader(file_path=str(pdf_path))
        docs = loader.load()
        print(f"[INFO] Loaded {len(docs)} pages from {pdf_path.name}")

        # 3. Split into chunks
        print("[INFO] Executing text splitting sequence...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=400
        )
        chunks = text_splitter.split_documents(documents=docs)
        print(f"[INFO] Created {len(chunks)} chunks.")

        # 4. Embed and store in the new collection with Rate Limiting
        print(f"[INFO] Writing to new Chroma collection: '{new_collection}'")
        embedding_models = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        # Initialize an empty vector store first
        vector_store = Chroma(
            collection_name=new_collection,
            embedding_function=embedding_models,
            persist_directory=str(CHROMA_PATH)
        )

        # Process chunks in smaller batches and pause between them
        batch_size = 5
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            print(f"[INFO] Processing batch {i//batch_size + 1}...")
            
            # Add the current batch to Chroma
            vector_store.add_documents(documents=batch)
            
            # Pause to avoid hitting the 429 quota limit
            time.sleep(3)

        # 5. Persist the new active collection name
        set_active_collection(new_collection)
        print(f"[SUCCESS] Indexing complete. Active collection is now '{new_collection}'.")
        return new_collection

    except FileNotFoundError:
        print(f"[ERROR] Could not locate {pdf_path}. Verify the file exists.")
        raise
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        raise


if __name__ == "__main__":
    index_document()