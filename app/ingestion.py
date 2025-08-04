import aiohttp, io, hashlib, pathlib
import pdfplumber
from tqdm import tqdm

_ARTIFACTS_DIR = pathlib.Path("artifacts")
_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

async def fetch_and_split(pdf_url: str) -> list[str]:
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(60)) as sess:
        async with sess.get(pdf_url) as r:
            r.raise_for_status()
            raw_pdf = io.BytesIO(await r.read())

    chunks: list[str] = []
    with pdfplumber.open(raw_pdf) as pdf:
        total = len(pdf.pages)
        for i, p in enumerate(tqdm(pdf.pages, desc="PDF→text"), start=1):
            text = (p.extract_text() or "").strip()
            if text:
                chunks.append(f"Page {i} of {total}\n{text}")

    merged = "\n\n".join(chunks)
    h = hashlib.sha1(pdf_url.encode("utf-8")).hexdigest()[:12]
    out_path = _ARTIFACTS_DIR / f"ingest_{h}.txt"
    out_path.write_text(merged, encoding="utf-8")

    return chunks
