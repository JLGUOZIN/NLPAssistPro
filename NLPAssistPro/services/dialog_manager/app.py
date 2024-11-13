from fastapi import FastAPI, Request
from pydantic import BaseModel
from manager.py import DialogManager

app = FastAPI()
dialog_manager = DialogManager()

class ContextIn(BaseModel):
    user_id: str
    intent: str
    entities: dict

class ContextOut(BaseModel):
    context: dict

@app.post('/update_context', response_model=ContextOut)
async def update_context(context_in: ContextIn):
    user_id = context_in.user_id
    intent = context_in.intent
    entities = context_in.entities
    dialog_manager.update_context(user_id, intent, entities)
    context = dialog_manager.get_context(user_id)
    return {'context': context}
