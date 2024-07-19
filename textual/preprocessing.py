import re
from collections import defaultdict
import tqdm as notebook_tqdm
from transformers import BertTokenizer, BertModel
import torch
from sklearn.model_selection import train_test_split
import pickle

def parse_data(lines):
    data = defaultdict(list)
    
    # Regular expression pattern to match the format
    pattern = re.compile(r"^(?P<type>\w+)\s+sent\s+\((?P<label>\d+)\):\s+(?P<sentence>.+)$")
    
    for line in lines:
        match = pattern.match(line)
        if match:
            entry = {
                'label': match.group('label'),
                'sentence': match.group('sentence')
            }
            data[match.group('type')].append(entry)
    
    return data

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Encode the data
#TMD uses CLS token for sentence embeddings
def encode_texts(texts):
    # Tokenize the texts
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # The last hidden state is the output of the model
    sentence_embeddings = outputs.last_hidden_state[:, 0, :]
    
    return sentence_embeddings

def encode_dataset(dataset):
    encoded_data = defaultdict(list)
    for type_, entries in dataset.items():
        sentences = [entry['sentence'] for entry in entries]
        embeddings = encode_texts(sentences)
        for entry, embedding in zip(entries, embeddings):
            encoded_data[type_].append({
                'label': entry['label'],
                'sentence': entry['sentence'],
                'embedding': embedding
            })
    return encoded_data

def split_data(data, test_size=0.1):
    train_data = defaultdict(list)
    test_data = defaultdict(list)
    
    for type_, entries in data.items():
        train_entries, test_entries = train_test_split(entries, test_size=test_size)
        train_data[type_].extend(train_entries)
        test_data[type_].extend(test_entries)
    
    return train_data, test_data

def save_data(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
data_file = 'data/imdb/imdb_bert.txt'
#read all the data
with open(data_file, 'r') as f:
    data = f.readlines()
    parsed_data = parse_data(data) 
    
print("data parsed")

train_data, test_data = split_data(parsed_data, test_size=0.1)
print("data split")

encoded_train_data = encode_dataset(train_data)
print("train data encoded")
encoded_test_data = encode_dataset(test_data)
print("test data encoded")

# Save the encoded data
save_data('data/textfooler/encoded_train_data.pkl', encoded_train_data)
print("train data saved")
save_data('data/textfooler/encoded_test_data.pkl', encoded_test_data)
print("test data saved")
