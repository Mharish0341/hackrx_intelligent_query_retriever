from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

from app.config import GOOGLE_API_KEY
from app.vector_db import as_retriever

SYSTEM_PROMPT = (
    "You are an expert assistant for insurance-policy questions.\n"
    "Answer only with information found in the Context.\n"
    "If the Context lacks the required detail, respond exactly: Information not found.\n"
    "Provide a brief yet comprehensive answer."
)

def build_chain(vector_store):
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
        convert_system_message_to_human=True,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "Context:\n{context}\n\n{input}"),
        ]
    )

    retriever = as_retriever(vector_store)
    history_aware_retriever = create_history_aware_retriever(model, retriever, prompt)
    qa_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain
