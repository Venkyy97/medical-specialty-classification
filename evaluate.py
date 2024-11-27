import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import pandas as pd

def evaluate_model(model, test_data, label_encoder, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluate the trained model
    
    Parameters:
    model: trained DistilBERT model
    test_data: tuple of (encoded_texts, labels)
    label_encoder: fitted LabelEncoder
    device: 'cuda' or 'cpu'
    """
    model.eval()
    predictions = []
    actual = []

    test_loader = DataLoader(
        TensorDataset(
            test_data[0]['input_ids'],
            test_data[0]['attention_mask'],
            test_data[1]
        ),
        batch_size=16
    )

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            actual.extend(labels.numpy())

    # Convert numeric predictions back to labels
    pred_labels = label_encoder.inverse_transform(predictions)
    actual_labels = label_encoder.inverse_transform(actual)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(actual_labels, pred_labels))

    return pred_labels, actual_labels