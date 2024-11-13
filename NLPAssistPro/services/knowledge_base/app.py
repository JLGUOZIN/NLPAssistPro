from fastapi import FastAPI, Request
from pydantic import BaseModel
from knowledge_base import KnowledgeBase

app = FastAPI()
kb = KnowledgeBase()

class QueryIn(BaseModel):
    question: str

class AnswerOut(BaseModel):
    answer: str

@app.post('/query_kb', response_model=AnswerOut)
async def query_kb(query_in: QueryIn):
    question = query_in.question
    answer = kb.query(question)
    if answer:
        return {'answer': answer}
    else:
        return {'answer': "I'm sorry, I don't have an answer for that."}
