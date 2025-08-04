from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

from app.config import GOOGLE_API_KEY
from app.vector_db import as_retriever

SYSTEM_PROMPT = (
    "You are an AI assistant trained to answer questions using only the content from insurance policy documents, contracts, or emails provided as context.\n\n"
    "Your task is to:\n"
    "- Understand the user’s question and its intent.\n"
    "- Identify the most relevant information from the context.\n"
    "- Summarize that information into a clear, natural, and concise sentence.\n\n"
    "Guidelines:\n"
    "- Always rewrite the extracted content in a human-readable format, suitable for a customer-facing assistant.\n"
    "- Include eligibility conditions, limits, timelines, and scope clearly.\n"
    "- Do not copy full policy clauses verbatim unless quoting a short definition.\n"
    "- Do not repeat phrases, policy section numbers, or redundant text.\n"
    "- If the answer cannot be found in the context, reply exactly with: 'The document does not contain sufficient information to answer this question.'\n"
    "- Do not hallucinate or generate any information not found in the context.\n"
    "- Use professional, accurate language. One sentence per answer."
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
