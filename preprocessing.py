import pandas as pd
import re
from collections import Counter
import numpy as np

class MedicalTextPreprocessor:
    def __init__(self, data_path, n_categories=5):
        """
        Initialize preprocessor with data path and number of top categories to keep
        """
        self.data_path = data_path
        self.n_categories = n_categories
        self.df = None
        self.top_categories = None
    
    def load_data(self):
        """
        Load the dataset and perform initial analysis
        """
        self.df = pd.read_csv(self.data_path)
        print(f"Initial shape: {self.df.shape}")
        print("\nInitial category distribution:")
        print("\nColumn names in dataset:")
        print(self.df.columns.tolist())
        print(self.df['medical_speciality'].value_counts().head())
        
    def select_top_categories(self):
        """
        Select top N categories by frequency
        """
        category_counts = self.df['medical_speciality'].value_counts()
        self.top_categories = category_counts.head(self.n_categories).index.tolist()
        self.df = self.df[self.df['medical_speciality'].isin(self.top_categories)]
        print(f"\nSelected categories: {self.top_categories}")
        print(f"Shape after category selection: {self.df.shape}")
    def split_data(self, test_size =0.2, val_size= 0.1):
        from sklearn.model_selection import train_test_split
        train_val, test = train_test_split(self.df, test_size=test_size,stratify=self.df['medical_speciality'], random_state= 22)
        train, val = train_test_split(train_val, test_size=val_size/(1-test_size), stratify=train_val['medical_speciality'], random_state= 220)
        return train, val, test