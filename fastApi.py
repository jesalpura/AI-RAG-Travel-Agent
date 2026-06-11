from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import create_memory, get_travel_agent
from build_vector_db import KNOWLEDGE_BASE_DIR, build_vector_db
from db import db

app = FastAPI(
    title="AI RAG Travel Agent API",
    version="1.0",
    description="HTTP API for chat, dashboard stats, PDF uploads, and FAISS rebuilds.",
)

DEFAULT_CORS_ORIGINS = "http://localhost:8501,http://localhost:3000"
allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

SessionState = tuple[Any, int]
_sessions: dict[str, SessionState] = {}


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str | None = Field(default=None, description="Omit to start a new session.")


class ChatResponse(BaseModel):
    session_id: str
    text: str
    locations: list[dict[str, Any]] = Field(default_factory=list)


def _safe_pdf_name(filename: str) -> str:
    name = Path(filename).name
    stem = Path(name).stem
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")
    return f"{safe_stem or 'uploaded_document'}.pdf"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "travel-agent-api"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if req.session_id and req.session_id in _sessions:
        memory, db_session_id = _sessions[req.session_id]
        session_key = req.session_id
    else:
        memory, db_session_id = create_memory()
        session_key = str(db_session_id)
        _sessions[session_key] = (memory, db_session_id)

    try:
        result = get_travel_agent(
            req.message,
            memory=memory,
            session_id=db_session_id,
        )
    except Exception as exc:
        db.log_error(str(exc), tool_name="FastAPI Chat", session_id=db_session_id, exc=exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        session_id=session_key,
        text=result.get("text", ""),
        locations=result.get("locations", []),
    )


@app.get("/stats")
def get_stats() -> dict[str, Any]:
    return {
        "stats": db.get_stats(),
        "tool_usage": db.get_tool_usage(),
        "top_cities": db.get_top_cities(),
        "daily_activity": db.get_daily_activity(),
    }


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)) -> dict[str, str]:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    safe_filename = _safe_pdf_name(filename)
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    destination = KNOWLEDGE_BASE_DIR / safe_filename
    destination.write_bytes(await file.read())

    return {"saved": safe_filename}


@app.post("/rebuild-db")
def rebuild_db(background_tasks: BackgroundTasks) -> dict[str, str]:
    background_tasks.add_task(build_vector_db)
    return {"status": "rebuilding in background"}

print(db.get_stats())