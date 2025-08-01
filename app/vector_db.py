import pathlib
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.config import TOP_K_RETRIEVE

_EMBED_MODEL = "models/text-embedding-004"
_INDEX_PATH  = pathlib.Path("faiss_store")

def _embeddings():
    return GoogleGenerativeAIEmbeddings(model=_EMBED_MODEL)

def build_or_load(chunks: list[str]) -> FAISS:
    if _INDEX_PATH.exists():
        return FAISS.load_local(str(_INDEX_PATH), _embeddings(),
                                allow_dangerous_deserialization=True)
    vs = FAISS.from_texts(chunks, _embeddings())
    vs.save_local(str(_INDEX_PATH))
    return vs

def as_retriever(vs: FAISS):
    return vs.as_retriever(search_kwargs={"k": TOP_K_RETRIEVE})
