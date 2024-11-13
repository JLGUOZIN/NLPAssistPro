import json
import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Replace URLs and emails
    text = re.sub(r'\S+@\S+', 'email', text)
    text = re.sub(r'http\S+|www\S+', 'url', text)
    # Replace numbers
    text = re.sub(r'\d+', 'number', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def load_and_preprocess():
    # Load dataset
    with open('data/multiwoz/data.json', 'r') as f:
        data = json.load(f)

    dialogs = []
    for dialog in data.values():
        for turn in dialog['log']:
            if 'text' in turn:
                dialogs.append({
                    'text': turn['text'],
                    'dialog_act': turn.get('dialog_act', {})
                })

    df = pd.DataFrame(dialogs)
    df['processed_text'] = df['text'].apply(preprocess_text)
    df.to_csv('data/processed_dialogs.csv', index=False)
    print("Data preprocessing completed.")

if __name__ == '__main__':
    load_and_preprocess()
