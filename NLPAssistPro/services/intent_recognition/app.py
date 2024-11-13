from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import torch.nn.functional as F

app = FastAPI()

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=12  # Update based on your intents
)
model.load_state_dict(torch.load('intent_model.pt'))
model.eval()

# Intent mappings
intents = [...]  # List your intents here
id_to_intent = {idx: intent for idx, intent in enumerate(intents)}

class TextIn(BaseModel):
    text: str

class IntentOut(BaseModel):
    intent: str
    confidence: float

@app.post('/predict_intent', response_model=IntentOut)
async def predict_intent(text_in: TextIn):
    text = text_in.text
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            token_type_ids=inputs['token_type_ids'],
            attention_mask=inputs['attention_mask']
        )
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
        intent = id_to_intent[predicted.item()]
        confidence = confidence.item()
    return {'intent': intent, 'confidence': confidence}
