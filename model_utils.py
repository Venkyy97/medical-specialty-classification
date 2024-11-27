from transformers import DistilBertTokenizer

def encode_data(texts, labels, tokenizer, max_length=128):
    encoded = tokenizer(texts, padding=True, truncation=True, 
                max_length=max_length, return_tensors='pt')
    return encoded, labels