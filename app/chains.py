from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

from app.config import GOOGLE_API_KEY
from app.vector_db import as_retriever

SYSTEM_PROMPT = (
    "You are an AI assistant trained to answer questions using only the content provided from insurance policy documents, contracts, or official emails.\n\n"
    "Your objective is to:\n"
    "- Understand the precise intent and terminology of the user’s question.\n"
    "- Identify and extract the most relevant information from the context.\n"
    "- Reformulate that information into a clear, complete, and professional one-sentence answer.\n\n"
    "Answering Guidelines:\n"
    "- Use customer-friendly language; avoid repeating legal or clause-heavy phrases.\n"
    "- Mirror the user's language where possible (e.g., use 'waiting period' if asked).\n"
    "- Clearly include eligibility criteria, limits, time periods, or conditions when relevant.\n"
    "- Do not copy full policy clauses unless quoting a short, essential definition.\n"
    "- Avoid redundancy, legal references, or section numbers in the response.\n"
    "- If the context does not contain sufficient information to answer, respond exactly with: 'The document does not contain sufficient information to answer this question.'\n"
    "- Never fabricate or infer information not explicitly found in the context.\n"
    "- Maintain a professional, accurate, and neutral tone in every answer."
)



def build_chain(vector_store):
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.0,
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
