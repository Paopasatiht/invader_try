from typing import Optional
from fastapi import FastAPI
from news_rag_agent import main as news_main
import asyncio

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/news_agent/{question}")
def read_item(question: str):
    agent_answer = asyncio.run(news_main(question))
    return agent_answer