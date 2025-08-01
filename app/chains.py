from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

from app.config import OPENAI_API_KEY
from app.vector_db import as_retriever

SYSTEM_PROMPT = """
Your name is AI Bot and you must act like an expert assistant and natural bot.

IMPORTANT INSTRUCTIONS
- Analyse Context + Input thoroughly before answering.
- Use only the retrieved Context; do not invent facts.
- Provide complete, professional answers; include all key numbers, limits, waiting periods, qualifying phrases.
- Be engaging yet concise; no offensive language.
- Prefer bullet / point-by-point formatting when it helps clarity.
- If the user query is unrelated to Context, reply: Information not found.

Chat History:
{chat_history}

Context:
{context}

Follow-up Input:
{input}

Helpful Answer:
"""

def build_chain(vector_store):
    model = ChatOpenAI(
        model_name="gpt-4.1",
        api_key=OPENAI_API_KEY,
        temperature=0.0,
    )

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder("chat_history"),
        ("user", "{input}"),
        ("user", "Given the conversation above, generate a search query that will retrieve the most relevant clauses for the last question.")
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    retriever = as_retriever(vector_store)

    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt,
    )

    qa_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain
