# Fake News Detection with Natural Language Processing

## Project Overview

This project implements a text classification system to identify whether a news headline is real or fake news. Using machine learning techniques, we built and compared multiple classifiers to distinguish between authentic and fabricated news headlines.

## Dataset

The project uses two datasets:
- *Training Data*: `dataset/training_data.csv` containing news headlines with labels (0 = fake news, 1 = real news)
- *Testing Data*: `dataset/testing_data.csv` with headlines to predict (labels initially marked as 2)

## Methodology

### Text Preprocessing
- Converting to lowercase
- Removing URLs, special characters, punctuation
- Removing stopwords
- Standardizing whitespace

### Feature Extraction
- **TF-IDF Vectorization**: With n-gram range (1,2), max features limited to 10,000
- **Bag of Words (Count Vectorization)**: As an alternative approach

### Models Tested
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Multinomial Naive Bayes
- XGBoost

### Hyperparameter Tuning
Grid search was performed on all models to optimize performance, testing various parameter combinations.

## Model Performance

### Best Performing Model
- **Model**: Support Vector Machine (SVM)
- **Accuracy**: 94.38%
- **AUC Score**: 0.9862

### Complete Model Comparison (Top Performers)
| Model | Accuracy | AUC Score |
|-------|----------|-----------|
| SVM | 94.38% | 0.9862 |
| Logistic Regression (Tuned) | 93.68% | 0.9843 |
| Multinomial Naive Bayes | 93.03% | 0.9813 |
| Random Forest | 91.20% | 0.9716 |
| XGBoost | 89.64% | 0.9681 |

## Implementation Details

The implementation follows these key steps:
1. Text preprocessing and cleaning
2. Feature extraction using TF-IDF and Bag of Words
3. Model training and evaluation
4. Hyperparameter tuning using grid search
5. Model comparison and selection
6. Final predictions on testing data

## Setup and Usage

### Prerequisites
- Python 3.x or Nix package manager
- Required packages (see `pyproject.toml`)

### Development Environment
For the best experience, this project uses Nix to manage dependencies:

```bash
nix develop
```

This will provide you with all necessary dependencies and tools.

### Installation
```bash
# If using Nix (recommended), enter the development environment
nix develop

# Or install required packages using uv directly
uv sync

# Or using pip with pyproject.toml
pip install -e .
```

### Files Generated
- `model_comparison_results.csv` - Performance metrics for all models
- `detailed_model_comparison.csv` - Detailed comparison of original vs tuned models
- Various visualization plots in the `visualizations/` directory
- Trained models saved in the `models/` directory

## Results

The best model (SVM) achieved high accuracy (94.38%) and excellent AUC score (0.9862), demonstrating strong performance in distinguishing between real and fake news headlines. The model successfully classified the test dataset, with results saved to `testing_data_with_predictions.csv`.

## Project Structure
```
nlp-project/
├── dataset/                 # Training and testing data
├── models/                  # Saved trained models
├── visualizations/          # Generated plots and charts
├── notebook.ipynb           # Main implementation notebook
├── notebook.py              # Python script version of the notebook
├── model_comparison_results.csv  # Model performance metrics
├── detailed_model_comparison.csv # Detailed model comparison
├── pyproject.toml           # Project dependencies and configuration
└── README.md
```
