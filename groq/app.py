import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time

from dotenv import load_dotenv
load_dotenv()

# Load Grow API Key
groq_api_key = os.environ['GROQ_API_KEY']

if "vector" not in st.session_state:
    st.session_state.embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.loader=WebBaseLoader("https://docs.smith.langchain.com/")
    st.session_state.docs = st.session_state.loader.load()

    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    st.session_state.final_docs=st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)

st.title("ChatGroq Demo")
llm = ChatGroq(api_key=groq_api_key, model="gemma2-9b-it")

prompt = ChatPromptTemplate.from_template(
    """Answer the questions based on the provided context only.
    Please provide the most accurate and precise, to-the-point response.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

document_chain = create_stuff_documents_chain(prompt=prompt, llm=llm)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

user_prompt = st.text_input("How can I help you today?", placeholder="Type any question about LangSmith")

if user_prompt:
    start = time.process_time()
    response = retrieval_chain.invoke({"input":user_prompt})
    end = time.process_time()
    print(f"Response Time: {end - start}")
    st.write(response['answer'])

    # With expander
    with st.expander("Document Similarity Search"):
        # Find relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-----------------------------------") 