import os
import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from indexingpipe import index_document
from retrivepipeline import ask_tutor

app = FastAPI(title="ChatPDF", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/upload-pdf")
async def upload_and_index(file: UploadFile = File(...)):
    
    if not file.filename.lower().endswith(".pdf"):
        return {"status": "error", "message": "Only PDF files are accepted."}

    original_stem = Path(file.filename).stem
    safe_name = f"tmp_{uuid.uuid4().hex}.pdf"
    file_path = os.path.join(os.getcwd(), safe_name)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        collection_name = index_document(file_name=file_path,display_name=file.filename)

        return {
            "status": "success",
            "message": f"'{file.filename}' has been indexed successfully.",
            "collection": collection_name
        }

    except RuntimeError as e:
        return {"status": "error", "message": str(e)}
    except Exception as e:
        return {"status": "error", "message": f"Failed to process document: {str(e)}"}
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/chat")
async def chat_with_tutor(question: str = Form(...)):
    if not question.strip():
        return {"status": "error", "answer": "Please provide a non-empty question."}
    try:
        result = ask_tutor(question)
        return {"status": "success", "answer": result["answer"], "pages": result["pages"]}
    except RuntimeError as e:
        return {"status": "error", "answer": str(e), "pages": []}
    except Exception as e:
        return {"status": "error", "answer": f"System error: {str(e)}", "pages": []}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)