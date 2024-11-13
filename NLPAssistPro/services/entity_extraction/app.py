from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
from transformers import BertTokenizerFast, BertForTokenClassification

app = FastAPI()

# Load tokenizer and model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForTokenClassification.from_pretrained(
    'bert-base-cased',
    num_labels=9  # Update based on your labels
)
model.load_state_dict(torch.load('ner_model.pt'))
model.eval()

# Label mappings
ids_to_labels = {...}  # Your label mappings

class TextIn(BaseModel):
    text: str

class EntitiesOut(BaseModel):
    entities: dict

@app.post('/extract_entities', response_model=EntitiesOut)
async def extract_entities(text_in: TextIn):
    text = text_in.text
    inputs = tokenizer(
        text.split(),
        is_split_into_words=True,
        return_tensors='pt'
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2)
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        labels = [ids_to_labels[pred.item()] for pred in predictions[0]]
        entities = {}
        for token, label in zip(tokens, labels):
            if label != 'O':
                entities.setdefault(label, []).append(token)
    return {'entities': entities}
