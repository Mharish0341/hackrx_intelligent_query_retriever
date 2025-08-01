# main.py  (project root)
from fastapi import FastAPI
from app.api import router          # <- path to the router inside /app

app = FastAPI(
    title="HackRx RAG Service",
    docs_url="/",                   # Swagger UI at /
    redoc_url=None
)

app.include_router(router)
