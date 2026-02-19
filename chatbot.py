import os
import traceback
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

# -------------------------------------------------------------------
# ENV LOADING (local development only)
# -------------------------------------------------------------------
if os.path.exists(".env"):
    load_dotenv(find_dotenv())

# -------------------------------------------------------------------
# APP CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="FinBot | Financial Intelligence Assistant",
    page_icon="📊",
    layout="centered"
)

# -------------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------------
DB_FAISS_PATH = "vectorstore/db_faiss"

CUSTOM_PROMPT_TEMPLATE = """
You are a Senior Equity Research Analyst.

Your task is to answer financial questions using ONLY the provided 10-K report context.

Guidelines:
- If numeric data is requested, carefully read table columns.
- Financial tables typically follow:
  [Item Name] [2025 Value] [2024 Value] [2023 Value]
- Clearly state if the requested year is missing.
- Do NOT hallucinate.
- Be concise, professional, and structured.

Context:
{context}

Question:
{question}

Answer:
"""

# -------------------------------------------------------------------
# VECTORSTORE LOADER
# -------------------------------------------------------------------
@st.cache_resource
def load_vectorstore():
    if not os.path.exists(DB_FAISS_PATH):
        st.error(f"FAISS index not found at: {DB_FAISS_PATH}")
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return FAISS.load_local(
        DB_FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

# -------------------------------------------------------------------
# LLM LOADER
# -------------------------------------------------------------------
def load_llm():
    groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        st.error("GROQ_API_KEY not found. Please configure Streamlit secrets.")
        st.stop()

    return ChatGroq(
        model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.2,
        groq_api_key=groq_api_key,
    )

# -------------------------------------------------------------------
# DOCUMENT FORMATTER
# -------------------------------------------------------------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -------------------------------------------------------------------
# BUILD RAG CHAIN
# -------------------------------------------------------------------
def build_rag_chain(vectorstore, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# -------------------------------------------------------------------
# MAIN UI
# -------------------------------------------------------------------
def main():
    st.title("📊 FinBot")
    st.caption("AI-Powered Financial Report Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a financial question from the 10-K report...")

    if user_input:
        # Display user message
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        try:
            with st.spinner("Analyzing financial report..."):
                vectorstore = load_vectorstore()
                llm = load_llm()
                rag_chain = build_rag_chain(vectorstore, llm)

                response = rag_chain.invoke(user_input)

            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response)

            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })

        except Exception:
            st.error("An unexpected error occurred:")
            st.code(traceback.format_exc())

# -------------------------------------------------------------------
# ENTRY POINT
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
