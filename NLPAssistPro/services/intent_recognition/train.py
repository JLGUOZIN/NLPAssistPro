import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
from tqdm import tqdm

# Load preprocessed data
df = pd.read_csv('../../data/processed_dialogs.csv')

# Define intents
intents = set()
for acts in df['dialog_act'].fillna('{}'):
    acts_dict = eval(acts)
    for act in acts_dict.keys():
        intents.add(act.split('-')[0])

intents = list(intents)
intent_to_id = {intent: idx for idx, intent in enumerate(intents)}
id_to_intent = {idx: intent for intent, idx in intent_to_id.items()}

# Map intents to labels
def get_intent_label(acts):
    if pd.isna(acts) or acts == '{}':
        return intent_to_id.get('general', 0)
    else:
        acts_dict = eval(acts)
        intent = list(acts_dict.keys())[0].split('-')[0]
        return intent_to_id.get(intent, intent_to_id.get('general', 0))

df['intent_label'] = df['dialog_act'].apply(get_intent_label)

# Split data
X_train, X_val, y_train, y_val = train_test_split(
    df['text'], df['intent_label'], test_size=0.1, random_state=42
)

# Dataset Class
class IntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer.encode_plus(
            self.texts[idx],
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(intents)
)

# Data Loaders
train_dataset = IntentDataset(X_train, y_train, tokenizer)
val_dataset = IntentDataset(X_val, y_val, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Training Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 3
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss()

# Training Function
def train_model():
    for epoch in range(3):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss}")
        evaluate_model()

def evaluate_model():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            logits = outputs.logits
            total_eval_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_eval_accuracy += torch.sum(preds == labels)
    avg_val_accuracy = total_eval_accuracy.double() / len(val_loader.dataset)
    avg_val_loss = total_eval_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss}, Validation Accuracy: {avg_val_accuracy}")

    # Save model after evaluation
    torch.save(model.state_dict(), 'intent_model.pt')
    print("Model saved.")

if __name__ == '__main__':
    train_model()
