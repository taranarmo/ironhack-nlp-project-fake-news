#!/usr/bin/env python

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available, skipping this model")
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import string
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# %%
# ## Load the Datasets

# Load training data
train_df = pd.read_csv(
    '/home/taranarmo/ironhack/week7/nlp-project/dataset/training_data.csv',
    header=None,
    sep='\t',
    names=['label', 'text']
)

# Load testing data
test_df = pd.read_csv(
    '/home/taranarmo/ironhack/week7/nlp-project/dataset/testing_data.csv',
    header=None,
    sep='\t',
    names=['label', 'text']
)

print("Training data shape:", train_df.shape)
print("Testing data shape:", test_df.shape)

# Display first few rows of training data
print("\nFirst 5 rows of training data:")
print(train_df.head())

print("\nTraining data info:")
print(train_df.info())

print("\nLabel distribution in training data:")
print(train_df['label'].value_counts())

print("\nFirst 5 rows of testing data:")
print(test_df.head())

print("\nLabel distribution in testing data (should be all 2s initially):")
print(test_df['label'].value_counts())

# ## Split the Training Data
# Split the training data into training and validation sets
X = train_df['text']
y = train_df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)

print(f"\nTraining set shape: {X_train.shape[0]} samples")
print(f"Validation set shape: {X_test.shape[0]} samples")
print(f"Training label distribution:\n{y_train.value_counts()}")
print(f"Validation label distribution:\n{y_test.value_counts()}")

# %%
# ## Text Preprocessing

def preprocess_text(text):
    """
    Preprocess text by:
    - Converting to lowercase
    - Removing punctuation and special characters
    - Removing extra whitespaces
    - Removing stopwords
    """
    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

    # Remove user @ references and '#' from tweet
    text = re.sub(r'\@\w+|\#', '', text)

    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespaces
    text = ' '.join(text.split())

    return text

# Apply preprocessing to training and validation sets
print("\nApplying preprocessing to training and validation sets...")
X_train_processed = X_train.apply(preprocess_text)
X_test_processed = X_test.apply(preprocess_text)

# Also preprocess the testing data
X_test_final = test_df['text'].apply(preprocess_text)

print("Preprocessing completed!")
print(f"Example of original text: {X_train.iloc[0]}")
print(f"Example of processed text: {X_train_processed.iloc[0]}")

# %%
# ## Feature Extraction using TF-IDF

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=10000,        # Limit to top 10k features
    ngram_range=(1, 2),        # Use unigrams and bigrams
    stop_words='english',      # Remove English stop words
    lowercase=True,            # Already handled in preprocessing
    min_df=2,                  # Ignore terms that appear in less than 2 documents
    max_df=0.95                # Ignore terms that appear in more than 95% of documents
)

# Fit the vectorizer on the training data and transform
print("\nApplying TF-IDF vectorization...")
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_processed)
X_test_tfidf = tfidf_vectorizer.transform(X_test_processed)

# Transform the final test data
X_test_final_tfidf = tfidf_vectorizer.transform(X_test_final)

print(f"Training TF-IDF shape: {X_train_tfidf.shape}")
print(f"Validation TF-IDF shape: {X_test_tfidf.shape}")
print(f"Final test TF-IDF shape: {X_test_final_tfidf.shape}")

# Also try with Count Vectorizer (Bag of Words)
count_vectorizer = CountVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words='english',
    lowercase=True,
    min_df=2,
    max_df=0.95
)

# Fit the count vectorizer on the training data and transform
print("\nApplying Bag of Words vectorization...")
X_train_bow = count_vectorizer.fit_transform(X_train_processed)
X_test_bow = count_vectorizer.transform(X_test_processed)
X_test_final_bow = count_vectorizer.transform(X_test_final)

print(f"Training BoW shape: {X_train_bow.shape}")
print(f"Validation BoW shape: {X_test_bow.shape}")
print(f"Final test BoW shape: {X_test_final_bow.shape}")

# %%
# ## Model Training and Evaluation

# Define models to try
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, probability=True),
    'Multinomial Naive Bayes': MultinomialNB()
}

# Conditionally add XGBoost if available
if XGBOOST_AVAILABLE:
    models['XGBoost'] = xgb.XGBClassifier(random_state=42, n_estimators=100)
    print("XGBoost model added to the models list")
else:
    print("XGBoost not available, skipping this model")

# Train and evaluate each model using TF-IDF features
results = {}

print("\nTraining and evaluating models with TF-IDF features...")
for name, model in models.items():
    print(f"\nTraining {name}...")

    # Train the model
    model.fit(X_train_tfidf, y_train)

    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]  # Probability of positive class

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    # Store results
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'auc_score': auc_score,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

    print(f"{name} - Accuracy: {accuracy:.4f}, AUC: {auc_score:.4f}")

# Ensure models directory exists
import os
os.makedirs('models', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

# Save all original models
print("\nSaving original models...")
for name, result in results.items():
    model_filename = f"models/original_{name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(result['model'], model_filename)
    print(f"Saved {name} model to {model_filename}")

# Also evaluate some models with Bag of Words features
print("\nEvaluating selected models with Bag of Words features...")

bow_results = {}

for name in ['Logistic Regression', 'Multinomial Naive Bayes']:
    if name in models:
        model = models[name]
        print(f"\nTraining {name} with BoW features...")

        # Train on BoW features
        model_bow = type(model)(**model.get_params())  # Create a new instance with same parameters
        model_bow.fit(X_train_bow, y_train)

        # Make predictions
        y_pred_bow = model_bow.predict(X_test_bow)
        y_pred_proba_bow = model_bow.predict_proba(X_test_bow)[:, 1]

        # Calculate metrics
        accuracy_bow = accuracy_score(y_test, y_pred_bow)
        auc_score_bow = roc_auc_score(y_test, y_pred_proba_bow)

        # Store results
        bow_results[name] = {
            'model': model_bow,
            'accuracy': accuracy_bow,
            'auc_score': auc_score_bow,
            'predictions': y_pred_bow,
            'probabilities': y_pred_proba_bow
        }

        print(f"{name} (BoW) - Accuracy: {accuracy_bow:.4f}, AUC: {auc_score_bow:.4f}")

# Save Bag of Words models
print("\nSaving Bag of Words models...")
for name, result in bow_results.items():
    model_filename = f"models/bow_{name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(result['model'], model_filename)
    print(f"Saved {name} (BoW) model to {model_filename}")

# %%
# ## Hyperparameter Tuning

print("\nPerforming hyperparameter tuning for best performing models...")

# Define parameter grids for hyperparameter tuning
param_grids = {
    'Logistic Regression': {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']  # Required for l1 penalty
    },
    'Random Forest': {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'SVM': {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    'Multinomial Naive Bayes': {
        'alpha': [0.1, 1.0, 10.0]
    }
}

# Add XGBoost to the grid if available
if XGBOOST_AVAILABLE:
    param_grids['XGBoost'] = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2]
    }

# Perform hyperparameter tuning for each model
tuned_results = {}

for name, params in param_grids.items():
    print(f"\nTuning hyperparameters for {name}...")

    # Create a new instance of the model with standard parameters
    if name == 'Logistic Regression':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif name == 'Random Forest':
        model = RandomForestClassifier(random_state=42)
    elif name == 'SVM':
        model = SVC(random_state=42, probability=True)
    elif name == 'Multinomial Naive Bayes':
        model = MultinomialNB()
    elif name == 'XGBoost' and XGBOOST_AVAILABLE:
        model = xgb.XGBClassifier(random_state=42)
    else:
        continue  # Skip if model not available

    # Perform grid search
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        cv=3,  # 3-fold cross-validation to save time
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    # Fit the grid search on the training data
    grid_search.fit(X_train_tfidf, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Make predictions on validation set
    y_pred_tuned = best_model.predict(X_test_tfidf)
    y_pred_proba_tuned = best_model.predict_proba(X_test_tfidf)[:, 1]

    # Calculate metrics
    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
    auc_score_tuned = roc_auc_score(y_test, y_pred_proba_tuned)

    # Store results
    tuned_results[name] = {
        'model': best_model,
        'accuracy': accuracy_tuned,
        'auc_score': auc_score_tuned,
        'best_params': grid_search.best_params_,
        'predictions': y_pred_tuned,
        'probabilities': y_pred_proba_tuned
    }

    print(f"{name} (Tuned) - Accuracy: {accuracy_tuned:.4f}, AUC: {auc_score_tuned:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")

# Save all tuned models
print("\nSaving tuned models...")
for name, result in tuned_results.items():
    model_filename = f"models/tuned_{name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(result['model'], model_filename)
    print(f"Saved {name} (Tuned) model to {model_filename}")

# %%
# ## Model Comparison

# Create a comparison dataframe
comparison_data = []

for name, result in tuned_results.items():
    comparison_data.append({
        'Model': name,
        'Accuracy': result['accuracy'],
        'AUC Score': result['auc_score'],
        'Type': 'Tuned',
        'Best_Params': str(result.get('best_params', 'N/A'))
    })

for name, result in results.items():
    comparison_data.append({
        'Model': name,
        'Accuracy': result['accuracy'],
        'AUC Score': result['auc_score'],
        'Type': 'Original',
        'Best_Params': 'N/A'
    })

# Also add Bag of Words results to comparison
for name, result in bow_results.items():
    comparison_data.append({
        'Model': f"{name} (BoW)",
        'Accuracy': result['accuracy'],
        'AUC Score': result['auc_score'],
        'Type': 'Bag of Words',
        'Best_Params': 'N/A'
    })

comparison_df = pd.DataFrame(comparison_data)
print("\nModel Comparison:")
print(comparison_df)

# Save comparison table to CSV
comparison_df.to_csv('model_comparison_results.csv', index=False)
print("\nModel comparison results saved to 'model_comparison_results.csv'")

# Create more detailed comparison tables
original_vs_tuned_comparison = []
for name in models.keys():
    if name in results and name in tuned_results:
        original_result = results[name]
        tuned_result = tuned_results[name]
        original_vs_tuned_comparison.append({
            'Model': name,
            'Original_Accuracy': original_result['accuracy'],
            'Tuned_Accuracy': tuned_result['accuracy'],
            'Original_AUC': original_result['auc_score'],
            'Tuned_AUC': tuned_result['auc_score'],
            'Accuracy_Improvement': tuned_result['accuracy'] - original_result['accuracy'],
            'AUC_Improvement': tuned_result['auc_score'] - original_result['auc_score'],
            'Best_Params': str(tuned_result.get('best_params', 'N/A'))
        })

if original_vs_tuned_comparison:
    comparison_detail_df = pd.DataFrame(original_vs_tuned_comparison)
    print("\nDetailed Model Comparison (Original vs Tuned):")
    print(comparison_detail_df)

    # Save detailed comparison to CSV
    comparison_detail_df.to_csv('detailed_model_comparison.csv', index=False)
    print("\nDetailed model comparison results saved to 'detailed_model_comparison.csv'")

# Find the best model based on AUC score
best_model_name = comparison_df.loc[comparison_df['AUC Score'].idxmax()]['Model']
best_model_type = comparison_df.loc[comparison_df['AUC Score'].idxmax()]['Type']
best_auc = comparison_df['AUC Score'].max()
best_accuracy = comparison_df.loc[comparison_df['AUC Score'].idxmax()]['Accuracy']

print(f"\nBest Model: {best_model_name} ({best_model_type})")
print(f"Best AUC Score: {best_auc:.4f}")
print(f"Best Accuracy: {best_accuracy:.4f}")

# Get the actual best model
if best_model_type == 'Tuned':
    best_model = tuned_results[best_model_name]['model']
else:
    best_model = results[best_model_name]['model']

# Detailed classification report for the best model
print(f"\nDetailed Classification Report for {best_model_name} ({best_model_type}):")
if best_model_type == 'Tuned':
    best_predictions = tuned_results[best_model_name]['predictions']
else:
    best_predictions = results[best_model_name]['predictions']

print(classification_report(y_test, best_predictions))

# Create model performance comparison visualizations
print("\nCreating model performance comparison visualizations...")

# Create a bar plot for accuracy comparison
plt.figure(figsize=(12, 8))
comparison_df_sorted = comparison_df.sort_values('Accuracy', ascending=True)
plt.subplot(2, 1, 1)
bars1 = plt.barh(range(len(comparison_df_sorted)), comparison_df_sorted['Accuracy'],
                 color=['red' if 'Original' in t else 'blue' if 'Tuned' in t else 'green'
                        for t in comparison_df_sorted['Type']])
plt.yticks(range(len(comparison_df_sorted)), comparison_df_sorted['Model'])
plt.xlabel('Accuracy')
plt.title('Model Accuracy Comparison')
for i, v in enumerate(comparison_df_sorted['Accuracy']):
    plt.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
plt.xlim(0, 1)

# Create a bar plot for AUC comparison
plt.subplot(2, 1, 2)
bars2 = plt.barh(range(len(comparison_df_sorted)), comparison_df_sorted['AUC Score'],
                 color=['red' if 'Original' in t else 'blue' if 'Tuned' in t else 'green'
                        for t in comparison_df_sorted['Type']])
plt.yticks(range(len(comparison_df_sorted)), comparison_df_sorted['Model'])
plt.xlabel('AUC Score')
plt.title('Model AUC Score Comparison')
for i, v in enumerate(comparison_df_sorted['AUC Score']):
    plt.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
plt.xlim(0, 1)

plt.tight_layout()
plt.savefig('visualizations/model_performance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Model performance comparison plot saved as 'visualizations/model_performance_comparison.png'")

# Create a detailed comparison plot for original vs tuned models
if 'comparison_detail_df' in locals():
    plt.figure(figsize=(14, 10))

    # Plot original vs tuned accuracy
    plt.subplot(2, 2, 1)
    x = np.arange(len(comparison_detail_df))
    width = 0.35
    plt.bar(x - width/2, comparison_detail_df['Original_Accuracy'], width, label='Original', alpha=0.8)
    plt.bar(x + width/2, comparison_detail_df['Tuned_Accuracy'], width, label='Tuned', alpha=0.8)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Original vs Tuned Model Accuracy')
    plt.xticks(x, comparison_detail_df['Model'], rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1)

    # Plot original vs tuned AUC
    plt.subplot(2, 2, 2)
    plt.bar(x - width/2, comparison_detail_df['Original_AUC'], width, label='Original', alpha=0.8)
    plt.bar(x + width/2, comparison_detail_df['Tuned_AUC'], width, label='Tuned', alpha=0.8)
    plt.xlabel('Model')
    plt.ylabel('AUC Score')
    plt.title('Original vs Tuned Model AUC')
    plt.xticks(x, comparison_detail_df['Model'], rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1)

    # Plot accuracy improvement
    plt.subplot(2, 2, 3)
    plt.bar(x, comparison_detail_df['Accuracy_Improvement'],
            color=['green' if imp > 0 else 'red' for imp in comparison_detail_df['Accuracy_Improvement']])
    plt.xlabel('Model')
    plt.ylabel('Accuracy Improvement')
    plt.title('Accuracy Improvement from Tuning')
    plt.xticks(x, comparison_detail_df['Model'], rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Plot AUC improvement
    plt.subplot(2, 2, 4)
    plt.bar(x, comparison_detail_df['AUC_Improvement'],
            color=['green' if imp > 0 else 'red' for imp in comparison_detail_df['AUC_Improvement']])
    plt.xlabel('Model')
    plt.ylabel('AUC Improvement')
    plt.title('AUC Improvement from Tuning')
    plt.xticks(x, comparison_detail_df['Model'], rotation=45, ha='right')
    plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('visualizations/original_vs_tuned_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Original vs Tuned comparison plot saved as 'visualizations/original_vs_tuned_comparison.png'")

# Confusion Matrix
try:
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Real News', 'Fake News'],
                yticklabels=['Real News', 'Fake News'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    # Save the plot instead of showing
    plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print("Confusion matrix plot saved as 'confusion_matrix.png'")
except Exception as e:
    print(f"Could not create confusion matrix plot: {e}")

# %%
# ## Final Predictions on Testing Data

print(f"\nUsing the best model ({best_model_name}) to predict labels for testing data...")

# Transform the testing data using the same TF-IDF vectorizer used for training
X_test_final_transformed = tfidf_vectorizer.transform(X_test_final)

# Make predictions on the testing data
final_predictions = best_model.predict(X_test_final_transformed)

print(f"Predicted labels shape: {final_predictions.shape}")
print(f"Predicted label distribution:\n{pd.Series(final_predictions).value_counts()}")

# Create a copy of the test dataframe to update
test_predictions_df = test_df.copy()

# Replace the labels (currently all 2s) with our predictions
test_predictions_df['label'] = final_predictions

print(f"\nUpdated testing data shape: {test_predictions_df.shape}")
print("First 10 rows of updated testing data:")
print(test_predictions_df.head(10))

# Save the updated testing data with predictions
output_file_path = 'testing_data_with_predictions.csv'
test_predictions_df.to_csv(output_file_path, sep='\t', header=False, index=False)
print(f"\nPredictions saved to: {output_file_path}")

print("\nFake News Classification Project completed!")
print(f"Best model: {best_model_name}")
print(f"Best AUC score: {best_auc:.4f}")
print(f"Predictions made on testing data and saved to {output_file_path}")
