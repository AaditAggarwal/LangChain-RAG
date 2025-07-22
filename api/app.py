from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="LangChain Local LLM",
    version="1.0",
    description="A FastAPI application using LangChain with Ollama for local LLM capabilities."
)

# Adding route for paid model
# add_routes(
#     app,
#     Ollama(),
#     path="/openai"
# )

model = Ollama(model="llama2")

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} in 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} in 100 words")

add_routes(
    app,
    prompt1 | model,
    path="/essay"
)

add_routes(
    app,
    prompt2 | model,
    path="/poem"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)