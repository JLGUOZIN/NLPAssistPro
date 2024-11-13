import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
from tqdm import tqdm

# Load and preprocess data in CoNLL format
# Assume data is prepared and saved as 'ner_dataset.csv' with columns 'sentence' and 'labels'

df = pd.read_csv('../../data/ner_dataset.csv')

# Map labels to IDs
labels = list(set(label for labels in df['labels'] for label in labels.split()))
labels_to_ids = {k: v for v, k in enumerate(labels)}
ids_to_labels = {v: k for k, v in labels_to_ids.items()}

# Dataset Class
class NERDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        word_labels = self.labels[idx].split()
        encoding = self.tokenizer(
            text.split(),
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        labels = [labels_to_ids[label] for label in word_labels]
        labels += [labels_to_ids['O']] * (self.max_len - len(labels))
        encoding['labels'] = torch.tensor(labels)
        return {key: val.squeeze() for key, val in encoding.items()}

# Tokenizer and Model
tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
model = BertForTokenClassification.from_pretrained(
    'bert-base-cased',
    num_labels=len(labels)
)

# Data Loaders
X_train, X_val, y_train, y_val = train_test_split(
    df['sentence'], df['labels'], test_size=0.1, random_state=42
)

train_dataset = NERDataset(X_train, y_train, tokenizer)
val_dataset = NERDataset(X_val, y_val, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Training Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=3e-5)
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
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}, Training Loss: {avg_train_loss}")
        evaluate_model()

def evaluate_model():
    model.eval()
    total_eval_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_eval_loss += loss.item()
    avg_val_loss = total_eval_loss / len(val_loader)
    print(f"Validation Loss: {avg_val_loss}")

    # Save model after evaluation
    torch.save(model.state_dict(), 'ner_model.pt')
    print("Model saved.")

if __name__ == '__main__':
    train_model()
