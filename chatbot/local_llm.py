from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# create iniitializing prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. You are funny too and try to lighten the mood."),
        ("user", "Question: {question}"),
    ]
)

# streamlit

st.title("Local LLM with Ollama")
input_text = st.text_input("How can I help you today?")

# Ollama
llm = Ollama(model="llama2", temperature=0.8)
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({"question": input_text}))