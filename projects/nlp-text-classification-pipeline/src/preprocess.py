"""
Data preprocessing module for NLP text classification pipeline.
Handles text cleaning, TF-IDF vectorization, and dataset preparation.
"""

import pandas as pd
import numpy as np
import re
import nltk
from typing import Tuple, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def preprocess_text(text: str, remove_stopwords: bool = True) -> str:
    """Clean and preprocess text for NLP tasks."""
    # Lowercase
    text = text.lower()
    # Remove special characters, keep alphanumeric and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    if remove_stopwords:
        try:
            stops = set(stopwords.words('english'))
            words = word_tokenize(text)
            words = [w for w in words if w not in stops and len(w) > 2]
            text = ' '.join(words)
        except:
            pass
    
    return text


def create_tfidf_features(texts: pd.Series, 
                         max_features: int = 5000,
                         ngram_range: Tuple[int, int] = (1, 2),
                         min_df: int = 3,
                         max_df: float = 0.95) -> Tuple[np.ndarray, TfidfVectorizer]:
    """Create TF-IDF features from text data."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        stop_words='english'
    )
    
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def create_count_features(texts: pd.Series, max_features: int = 5000) -> Tuple[np.ndarray, CountVectorizer]:
    """Create count-based features from text data."""
    vectorizer = CountVectorizer(
        max_features=max_features,
        stop_words='english',
        min_df=3
    )
    
    X = vectorizer.fit_transform(texts)
    return X, vectorizer


def prepare_train_test_split(df: pd.DataFrame, 
                            text_col: str = 'text_clean',
                            label_col: str = 'label',
                            test_size: float = 0.2,
                            random_state: int = 42) -> Tuple:
    """Split data into train/test sets with stratification."""
    from sklearn.model_selection import train_test_split
    
    X = df[text_col]
    y = df[label_col]
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def encode_labels(df: pd.DataFrame, category_col: str = 'category') -> Tuple[pd.DataFrame, dict]:
    """Encode categorical labels to integers."""
    label_map = {cat: idx for idx, cat in enumerate(df[category_col].unique())}
    df['label'] = df[category_col].map(label_map)
    return df, label_map


def get_class_weights(y: np.ndarray) -> dict:
    """Compute balanced class weights for imbalanced datasets."""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return {cls: weight for cls, weight in zip(classes, weights)}


def save_processed_data(df: pd.DataFrame, output_path: str):
    """Save processed dataframe."""
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to: {output_path}")


if __name__ == '__main__':
    # Test preprocessing
    test_text = "This is a sample government document about budget allocation and policy implementation."
    cleaned = preprocess_text(test_text)
    print(f"Original: {test_text}")
    print(f"Cleaned: {cleaned}")
    
    # Test TF-IDF
    texts = pd.Series(["budget document", "policy notice", "service request", "foia request"])
    X, vectorizer = create_tfidf_features(texts, max_features=10)
    print(f"TF-IDF shape: {X.shape}")
    print("Preprocessing module ready.")
