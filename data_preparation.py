def prepare_data_for_model(preprocessor):
    """_summary_
    Prepare the data for model training
    Args:
        preprocessor (_type_): _description_
    """
    from transformers import DistilBertTokenizer
    from sklearn.preprocessing import LabelEncoder
    import torch
    train_df, val_df, test_df = preprocessor.split_data()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    # Convert transcriptions to list and ensure they are strings
    train_texts = train_df['transcription'].astype(str).tolist()
    val_texts = val_df['transcription'].astype(str).tolist()
    test_texts = test_df['transcription'].astype(str).tolist()    
    label_encoder = LabelEncoder()
    train_labels = label_encoder.fit_transform(train_df['medical_speciality'])
    val_labels = label_encoder.transform(val_df['medical_speciality'])
    test_labels = label_encoder.transform(test_df['medical_speciality'])
    
    train_encodings = tokenizer(train_df['transcription'].astype(str).tolist(),
                                truncation = True,
                                padding = True,
                                return_tensors = 'pt')
    val_encodings = tokenizer(val_df['transcription'].astype(str).tolist(),
                                truncation = True,
                                padding = True,
                                return_tensors = 'pt')
    test_encodings = tokenizer(test_df['transcription'].astype(str).tolist(),
                                truncation = True,
                                padding = True,
                                return_tensors = 'pt')
    
    return (train_encodings, torch.tensor(train_labels)), \
           (val_encodings, torch.tensor(val_labels)), \
           (test_encodings, torch.tensor(test_labels)), \
           label_encoder