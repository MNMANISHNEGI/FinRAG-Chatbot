import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from dotenv import load_dotenv, find_dotenv

if os.path.exists(".env"):
    load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
    return prompt

def main():
    st.set_page_config(page_title="finBot Assistance")
    st.title("finBot: fincal Information Portal")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

    prompt = st.chat_input("Ask a fincal question...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        CUSTOM_PROMPT_TEMPLATE = """
ROLE: Senior Equity Research Analyst for NVIDIA.
TASK: Extract specific data from the 2025 10-K Annual Report.

REASONING FRAMEWORK:
1. Identify if the query is a numeric request (e.g., Revenue, Cash Flow) or qualitative (e.g., Risks).
2. For numeric requests:
   - Financial tables in 10-Ks are ordered as: [Item Name] [2025 Value] [2024 Value] [2023 Value].
   - Locate the exact row for the requested item.
   - Count the columns carefully: 2025 is the 1st value, 2024 is the 2nd, and 2023 is the 3rd.
3. If you do not see three columns of data, look for a header starting with 'Three Years Ended'.

CONSTRAINTS:
- Use only the provided context.
- Citation: Include the Page Number from the metadata.
- If data is missing for a specific year, state it clearly.

Context: {context}
Question: {question}

Analysis:"""

# Detailed Financial Analysis:"""
#         CUSTOM_PROMPT_TEMPLATE = """
#         Use the provided fincal context to answer the user's question.
#         If the answer is not in the context, clearly state that the information is not available in the provided document.
#         Keep the answer professional, structured, and easy to understand.

#         Context: {context}
#         Question: {question}

#         Answer:"""
        try:
            vectorstore = get_vectorstore()
        
            groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        
            if not groq_api_key:
                st.error("GROQ_API_KEY not found. Please check your .env or Streamlit secrets.")
                st.stop()
        
            from langchain_core.runnables import RunnablePassthrough
            from langchain_core.output_parsers import StrOutputParser
        
            retriever = vectorstore.as_retriever(search_kwargs={'k': 6})
        
            llm = ChatGroq(
                model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                temperature=0.2,
                groq_api_key=groq_api_key,
            )
        
            prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
        
            rag_chain = (
                {
                    "context": retriever,
                    "question": RunnablePassthrough(),
                }
                | prompt_template
                | llm
                | StrOutputParser()
            )
        
            result = rag_chain.invoke(prompt)
        
            with st.chat_message('assistant'):
                st.markdown(result)
        
            st.session_state.messages.append({'role': 'assistant', 'content': result})
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

       

if __name__ == "__main__":
    main()
