from transformers import DistilBertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tqdm import tqdm

def train_model(train_data, val_data, num_labels, epochs=3, 
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',num_labels = num_labels
    ).to(device)
    
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    train_dataset = TensorDataset(
        train_data[0]['input_ids'],
        train_data[0]['attention_mask'],
        train_data[1]
    )
    val_dataset = TensorDataset(
        val_data[0]['input_ids'],
        val_data[0]['attention_mask'],
        val_data[1]
   
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size= 16,
        shuffle= True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size= 16,
        shuffle= False
        
    )
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc = f'Epoch {epoch+1}/{epochs}'):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            optimizer.zero_grad
            
            outputs = model(
                input_ids = input_ids,
                attention_mask = attention_mask,
                labels = labels
            )
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
        avg_train_loss = total_train_loss/ len(train_loader)
        
        model.eval()
        total_val_loss = 0
        predictions = []
        acutal_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                outputs = model(
                    input_ids =input_ids,
                    attention_mask = attention_mask,
                    labels = labels
                )
                total_val_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim = 1)
                predictions.extend(preds.cpu().numpy())
                acutal_labels.extend(labels.cpu().numpy())
            avg_val_loss = total_val_loss / len(val_loader)
            print(f'\nEpoch {epoch+1}:')
            print(f'Average Training Loss: {avg_train_loss:.4f}')
            print(f'Average Validation Loss: {avg_val_loss:.4f}')
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), 'best_model.pt')
            return model
    