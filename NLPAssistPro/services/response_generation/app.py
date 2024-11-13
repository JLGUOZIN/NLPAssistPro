from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = FastAPI()

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('./response_model')
model = GPT2LMHeadModel.from_pretrained('./response_model')
model.eval()

class ContextIn(BaseModel):
    context: str

class ResponseOut(BaseModel):
    response: str

@app.post('/generate_response', response_model=ResponseOut)
async def generate_response(context_in: ContextIn):
    context_text = context_in.context
    inputs = tokenizer.encode(context_text, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_length=150,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2,
        num_beams=5,
        early_stopping=True
    )
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract agent's response
    response = response_text.split('Agent:')[-1].strip()
    return {'response': response}
