# ChatPDF — RAG PDF Tutor (FastAPI + LangChain + ChromaDB)

ChatPDF is a **Retrieval-Augmented Generation (RAG)** system that turns a PDF into a queryable knowledge base.
It supports:

- **Indexing**: PDF → pages → chunks → embeddings → **ChromaDB** collection
- **Tutoring**: user question → similarity search → cited context → grounded answer

This README is written to be **interview-friendly**: it explains design decisions, data flow, and API contracts.

## Tech stack

- **FastAPI**: HTTP API (`/upload-pdf`, `/chat`)
- **LangChain**:
  - `PyPDFLoader` for page-level parsing
  - `RecursiveCharacterTextSplitter` for chunking
  - `Chroma` (`langchain-chroma`) for vector persistence + retrieval
  - `ChatOpenAI` for answer generation
- **ChromaDB**: local, embedded vector DB (persisted to disk, no separate server required)
- **OpenAI embeddings**: `text-embedding-3-large` (dimension inferred by LangChain)
- **LLM**: `gpt-4o-mini` (low-latency tutoring)

## Architecture (high-level)

**Frontend (static)**

- `index.html` + `style.css` + `script.js`
- Calls FastAPI directly over HTTP

**Backend (FastAPI)**

- `main.py`
  - uploads PDF
  - invokes the indexing pipeline
  - exposes the chat endpoint

**Indexing pipeline**

- `indexingpipe.py`
  1. Load PDF pages
  2. Split into chunks
  3. Embed chunks
  4. Write vectors to a ChromaDB collection (persisted locally)
  5. Persist "active collection" to `.active_collection`

**Retrieval + generation pipeline**

- `retrivepipeline.py`
  - lazy-connects to the active ChromaDB collection
  - similarity search for context
  - LLM generation grounded in retrieved context
  - returns **answer + cited page numbers**

## Data flow (sequence)

### 1) Indexing flow

1. `POST /upload-pdf` (multipart PDF)
2. Server stores upload to a safe temp file (`tmp_<uuid>.pdf`)
3. `index_document(file_name=<tmp.pdf>)`:
   - deletes previous collection (single-active-document design)
   - loads pages via `PyPDFLoader`
   - chunking via `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)`
   - embeddings via `OpenAIEmbeddings(text-embedding-3-large)`
   - persists to a ChromaDB collection derived from filename
4. Active collection name persisted to `.active_collection`
5. Temp file removed

### 2) Chat flow

1. `POST /chat` with form field `question`
2. Backend runs `ask_tutor(question)`:
   - `similarity_search(k=4)` against active collection
   - builds a strict "answer using ONLY context" system prompt
   - calls the LLM
3. Returns:
   - `answer` (no page numbers embedded in text)
   - `pages` (deduped list, derived from chunk metadata)

## Local setup

### 1) Environment variables

```bash
cd ChatPDF
cp .env.example .env
```

Set:

- `GOOGLE_API_KEY=...`

### 2) Install dependencies

If you want a single environment for the whole repo:

```bash
pip install -r requirements.txt
```

Or install only ChatPDF deps:

```bash
pip install -r ChatPDF/requirements.txt
```

### 3) Run the API

```bash
cd ChatPDF
python main.py
```

Default: `http://127.0.0.1:8000`

ChromaDB runs embedded in-process and persists its data to a local directory (e.g. `./chroma_db`) — no separate database service needs to be started.

### 4) Run the UI

Open `ChatPDF/index.html` in the browser.

## API contracts (technical)

### `POST /upload-pdf`

**Content-Type**: `multipart/form-data`

- field: `file` (PDF)

Response (success):

```json
{
  "status": "success",
  "message": "'MyBook.pdf' has been indexed successfully.",
  "collection": "mybook"
}
```

### `POST /chat`

**Content-Type**: `multipart/form-data`

- field: `question` (string)

Response (success):

```json
{
  "status": "success",
  "answer": "...",
  "pages": ["12", "13"]
}
```

## Key design decisions (what to highlight in interviews)

1. **Single-active-document collections**
   - The system keeps one "active" ChromaDB collection tracked via `.active_collection`.
   - Pros: simple mental model, no multi-tenant routing, low ops overhead.
   - Tradeoff: multi-document support would require collection routing per user/session.

2. **Chunking strategy**
   - `chunk_size=1000`, `chunk_overlap=400` balances retrieval recall vs token budget.
   - Overlap reduces boundary information loss (definitions spanning pages/sections).

3. **Grounded answering**
   - System prompt explicitly restricts the model to provided context.
   - Page references are returned separately to avoid "hallucinated citations".

4. **Safe upload handling**
   - Uploaded PDFs are stored as `tmp_<uuid>.pdf` to avoid name collisions and path traversal.

## Troubleshooting

- **`No document indexed yet`**
  - Upload a PDF first (indexing must run before chat).

- **`Collection '<name>' not found in ChromaDB`**
  - The local Chroma persistence directory may have been deleted or reset. Re-upload the PDF.

- **Permission / disk errors writing to the Chroma persistence directory**
  - Ensure the process has write access to the configured persist directory (e.g. `./chroma_db`).

## Repo notes

- ChromaDB persistence directory is configured in `indexingpipe.py` (embedded, on-disk — no external DB server or Docker required).
- `.active_collection` is runtime state and should not be committed.
- Use `.env.example` and don't commit secrets.