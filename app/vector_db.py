import pathlib
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from app.config import TOP_K_RETRIEVE

_EMBED_MODEL  = "models/text-embedding-004"
_INDEX_ROOT   = pathlib.Path("faiss_store")
_INDEX_ROOT.mkdir(parents=True, exist_ok=True)

def _embeddings():
    return GoogleGenerativeAIEmbeddings(model=_EMBED_MODEL)

def _index_path_for(chunks: List[str]) -> pathlib.Path:
    # Single, fixed FAISS index (no hashing)
    return _INDEX_ROOT / "default"

def build_or_load(chunks: List[str]) -> FAISS:
    index_path = _index_path_for(chunks)
    if index_path.exists():
        return FAISS.load_local(str(index_path), _embeddings(), allow_dangerous_deserialization=True)

    docs: List[Document] = []
    for idx, page_content in enumerate(chunks, start=1):
        docs.append(
            Document(
                page_content=page_content,
                metadata={"chunk_id": idx}  # carries the page number since each chunk = a page
            )
        )

    vs = FAISS.from_documents(docs, _embeddings())
    vs.save_local(str(index_path))
    return vs

def as_retriever(vs: FAISS):
    return vs.as_retriever(search_kwargs={"k": TOP_K_RETRIEVE})
