# Medical Specialty Classification using DistilBERT: Automated Analysis of Clinical Transcriptions

## Project Overview
An automated medical text classification system that categorizes clinical transcriptions into different medical specialties using DistilBERT, a lightweight transformer model. The system demonstrates practical application of NLP in healthcare domain for efficient document routing and organization.

## Features
- Automated classification of medical transcriptions into 5 specialties
- Fine-tuned DistilBERT model for medical domain
- Preprocessing pipeline for clinical text
- Performance metrics and evaluation
- Support for multi-class classification

## Technical Details
- **Model**: DistilBERT (distilbert-base-uncased)
- **Framework**: PyTorch, Transformers
- **Performance**:
  - Overall Accuracy: 67%
  - Surgery Specialty: F1-score 0.82, Recall 0.95
  - Consultations: F1-score 0.79, Recall 0.99
  - Radiology: F1-score 0.60

## Project Structure
```
medical_symptom_classifier/
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── data_preparation.py
│   ├── evaluate.py
│   └── main.py
├── data/
│   └── mtsamples.csv
├── models/
└── results/
```

## Requirements
- Python 3.8+
- PyTorch
- Transformers
- Pandas
- Scikit-learn
- NumPy

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python src/main.py
```

## Model Performance
- Handles various medical specialties including Surgery, Cardiovascular/Pulmonary, Radiology
- Best performance in Surgery and Consultation classifications
- Moderate performance in Radiology
- Areas for improvement in Orthopedic and Cardiovascular specialties

## Future Improvements
- Implement data augmentation for underrepresented classes
- Add support for more medical specialties
- Enhance preprocessing for medical terminology
- Improve model performance on minority classes

## License
MIT License

## Author
[Your Name]

Citations:
[1] https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/33258223/b40a3b95-3c11-4fdf-9da2-b8e6981531ef/paste.txt
