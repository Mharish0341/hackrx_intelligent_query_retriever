from fastapi import APIRouter, Header, HTTPException
from app.schemas import QueryRequest, QueryResponse
from app.config import TEAM_TOKEN
from app.ingestion import fetch_and_split
from app.vector_db import build_or_load
from app.chains import build_chain

router = APIRouter()

@router.post("/hackrx/run", response_model=QueryResponse)
async def hackrx_run(
    payload: QueryRequest,
    authorization: str = Header(...)
):
    if authorization != f"Bearer {TEAM_TOKEN}":
        raise HTTPException(401, detail="Invalid Bearer token")

    chunks = await fetch_and_split(str(payload.documents))
    vs     = build_or_load(chunks)
    rag    = build_chain(vs)

    answers = []
    for q in payload.questions:
        raw = rag.invoke({"input": q, "chat_history": []})
        text = raw["answer"] if isinstance(raw, dict) else raw
        clean = " ".join(text.split())         # ← remove \n  \t  double-spaces
        answers.append(clean)

    return QueryResponse(answers=answers)
