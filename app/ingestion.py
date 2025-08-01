import aiohttp, io, re
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.config import CHUNK_SIZE, CHUNK_OVERLAP
from tqdm import tqdm

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""]
)

async def fetch_and_split(pdf_url: str) -> list[str]:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(60)) as sess:
        async with sess.get(pdf_url) as r:
            r.raise_for_status()
            raw_pdf = io.BytesIO(await r.read())
    pages = []
    with pdfplumber.open(raw_pdf) as pdf:
        for p in tqdm(pdf.pages, desc="PDF→text"):
            pages.append(p.extract_text() or "")

    merged = re.sub(r"[ \t]+", " ", "\n".join(pages))
    return _splitter.split_text(merged)
