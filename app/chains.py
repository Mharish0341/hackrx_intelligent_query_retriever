from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain

from app.config import OPENAI_API_KEY
from app.vector_db import as_retriever

def build_chain(vector_store):
    model = ChatOpenAI(
        model_name="gpt-4.1",
        openai_api_key=OPENAI_API_KEY,
        temperature=0.0,
    )

    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the conversation above, generate a search query to look up information relevant to the last question.")
    ])

    qa_system_prompt = """
    You are an insurance-policy assistant.

    Objective  
    • Give clear, formal, user-friendly answers that mirror the style of the sample answers.

    Guidelines  
    1. **Rephrase without copying** – avoid stiff legal phrases like “shall be excluded”; prefer “is provided / is covered / there is a waiting period”.  
    2. **Keep only essential facts** – include key numbers, limits, waiting periods, eligibility rules. Skip minor sub-clauses unless crucial.  
    3. **Positive wording** – turn negatives into positives (e.g. “there is a waiting period of 36 months” rather than “expenses are excluded for 36 months”).

    Reference Style  
    • Waiting-period answer: “There is a waiting period of thirty-six (36) months …”.  
    • Maternity answer: “Yes, the policy covers maternity expenses, including childbirth … The benefit is limited to two deliveries …”.  
    • Hospital definition: “A hospital is defined as an institution with at least 10 inpatient beds … maintains daily records of patients.”

    Apply the same tone, brevity, and structure to the question and context provided below.
    """

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        ("human", "Context:\n{context}\n\nQuestion:\n{input}\n\nAnswer:"),
    ])

    retriever = as_retriever(vector_store)
    history_aware_retriever = create_history_aware_retriever(
        llm=model,
        retriever=retriever,
        prompt=retriever_prompt
    )

    qa_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    return rag_chain
