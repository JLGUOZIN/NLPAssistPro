from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests

app = FastAPI()

class UserInput(BaseModel):
    user_id: str
    text: str

@app.post('/chat')
async def chat(user_input: UserInput):
    user_id = user_input.user_id
    text = user_input.text

    # Step 1: Intent Recognition
    intent_response = requests.post(
        'http://localhost:8001/predict_intent',
        json={'text': text}
    ).json()
    intent = intent_response['intent']

    # Step 2: Entity Extraction
    entity_response = requests.post(
        'http://localhost:8002/extract_entities',
        json={'text': text}
    ).json()
    entities = entity_response['entities']

    # Step 3: Update Dialog Context
    context_response = requests.post(
        'http://localhost:8003/update_context',
        json={
            'user_id': user_id,
            'intent': intent,
            'entities': entities
        }
    ).json()
    context = context_response['context']

    # Step 4: Generate Response
    # Combine context into a string
    context_text = f"User: {text} Context: {context}"
    response_generation = requests.post(
        'http://localhost:8004/generate_response',
        json={'context': context_text}
    ).json()
    response = response_generation['response']

    return {'response': response}
