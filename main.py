import torch
import pickle
import os
from preprocessing import MedicalTextPreprocessor
from data_preparation import prepare_data_for_model
from train import train_model
from evaluate import evaluate_model
def main():
    """
    Main function to run the medical text classification pipeline
    """
    try:
        # Initialize preprocessor
        print("\n1. Initializing preprocessor and loading data...")
        preprocessor = MedicalTextPreprocessor('data/mtsamples.csv')
        preprocessor.load_data()
        preprocessor.select_top_categories()

        # Prepare data
        print("\n2. Preparing data for model...")
        train_data, val_data, test_data, label_encoder = prepare_data_for_model(preprocessor)
        
        print("\n3. Saving label encoder...")
        with open('models/label_encoder.pkl', 'wb') as f:
            pickle.dump(label_encoder, f)
        num_labels = len(preprocessor.top_categories)
        print(f"\nNumber of categories: {num_labels}")
        print("Categories:", preprocessor.top_categories)           
        print("\n4. Training model...")
        model = train_model(
            train_data=train_data,
            val_data=val_data,
            num_labels=num_labels,
            epochs=3
        )

        # Evaluate model
        print("\n5. Evaluating model...")
        predictions, actual = evaluate_model(
            model=model,
            test_data=test_data,
            label_encoder=label_encoder
        )

        # Save results
        print("\n6. Saving results...")
        import pandas as pd
        results_df = pd.DataFrame({
            'Actual': actual,
            'Predicted': predictions
        })
        results_df.to_csv('results/classification_results.csv', index=False)
        
        print("\nPipeline completed successfully!")
        return model, test_data, label_encoder

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Run the main pipeline
    model, test_data, label_encoder = main()