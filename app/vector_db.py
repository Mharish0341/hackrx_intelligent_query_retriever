import shutil
import tempfile
import pathlib
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from app.config import TOP_K_RETRIEVE

_EMBED_MODEL = "models/text-embedding-004"
_INDEX_ROOT = pathlib.Path("faiss_store")
_INDEX_ROOT.mkdir(parents=True, exist_ok=True)

def _embeddings():
    return GoogleGenerativeAIEmbeddings(model=_EMBED_MODEL)

def _index_path_for(_: List[str]) -> pathlib.Path:
    return _INDEX_ROOT / "default"

def _build_index_from_chunks(chunks: List[str]) -> FAISS:
    docs: List[Document] = [
        Document(page_content=page_content, metadata={"chunk_id": idx})
        for idx, page_content in enumerate(chunks, start=1)
    ]
    return FAISS.from_documents(docs, _embeddings())

def build_or_load(chunks: List[str]) -> FAISS:
    index_path = _index_path_for(chunks)
    vs = _build_index_from_chunks(chunks)
    tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="faiss_tmp_", dir=str(_INDEX_ROOT)))
    vs.save_local(str(tmp_dir))
    if index_path.exists():
        shutil.rmtree(index_path)
    shutil.move(str(tmp_dir), str(index_path))
    return vs

def as_retriever(vs: FAISS):
    return vs.as_retriever(search_kwargs={"k": TOP_K_RETRIEVE})
