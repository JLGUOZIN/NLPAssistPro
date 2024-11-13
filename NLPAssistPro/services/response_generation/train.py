import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Load conversation data
df = pd.read_csv('../../data/processed_dialogs.csv')

# Prepare data with input-output pairs
df['input_text'] = 'User: ' + df['text'].shift(1).fillna('') + ' Agent: ' + df['text']
df = df.dropna(subset=['input_text'])

# Dataset Class
class ResponseDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=512):
        self.texts = texts.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids
        }

# Tokenizer and Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Data Loaders
dataset = ResponseDataset(df['input_text'], tokenizer)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# Training Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training Loop
def train_model():
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"Training Epoch {epoch+1}"):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")

    # Save model
    model.save_pretrained('./response_model')
    tokenizer.save_pretrained('./response_model')
    print("Model saved.")

if __name__ == '__main__':
    train_model()
