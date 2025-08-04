
from fastapi import FastAPI
from app.api import router as api_router

app = FastAPI(
    title="HackRx RAG Service",
    docs_url="/",   # Swagger UI at /
    redoc_url=None
)
app.include_router(api_router)
