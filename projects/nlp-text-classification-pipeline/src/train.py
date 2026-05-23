"""
Model training module for NLP text classification.
Supports Logistic Regression, BERT fine-tuning, and LLM zero-shot classification.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, List, Optional, Tuple
import pickle


class TextClassifier:
    """Text classification model wrapper."""
    
    def __init__(self, model_type: str = 'logistic_regression'):
        self.model_type = model_type
        self.model = None
        self.label_map = None
        self.fitted = False
        
    def fit(self, X: np.ndarray, y: np.ndarray, 
            label_map: Optional[Dict] = None,
            class_weights: Optional[Dict] = None):
        """Train classification model."""
        self.label_map = label_map
        
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                C=1.0,
                class_weight=class_weights or 'balanced'
            )
            self.model.fit(X, y)
            self.fitted = True
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.fitted:
            raise ValueError("Model not fitted.")
        return self.model.predict_proba(X)
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
    
    def get_top_features(self, feature_names: List[str], n: int = 10) -> Dict:
        """Get top features per class."""
        if self.model_type != 'logistic_regression':
            return {}
        
        top_features = {}
        for i, class_name in enumerate(self.model.classes_):
            if self.label_map:
                class_name = [k for k, v in self.label_map.items() if v == class_name][0]
            
            coef = self.model.coef_[i]
            top_idx = np.argsort(coef)[-n:][::-1]
            top_features[class_name] = [(feature_names[idx], coef[idx]) for idx in top_idx]
        
        return top_features
    
    def save(self, path: str):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_map': self.label_map,
                'model_type': self.model_type
            }, f)
        print(f"Saved model to: {path}")
    
    def load(self, path: str):
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.label_map = data['label_map']
        self.model_type = data['model_type']
        self.fitted = True
        print(f"Loaded model from: {path}")


def train_bert_classifier(train_texts: List[str], train_labels: List[int],
                          val_texts: Optional[List[str]] = None,
                          val_labels: Optional[List[int]] = None,
                          model_name: str = 'bert-base-uncased',
                          num_epochs: int = 3) -> Dict:
    """Fine-tune BERT for text classification."""
    try:
        from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
        import torch
        
        # Tokenize
        tokenizer = BertTokenizer.from_pretrained(model_name)
        
        def tokenize(texts):
            return tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors='pt')
        
        train_encodings = tokenize(train_texts)
        
        # Dataset
        class DocumentDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item
            
            def __len__(self):
                return len(self.labels)
        
        train_dataset = DocumentDataset(train_encodings, train_labels)
        
        # Model
        num_labels = len(set(train_labels))
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        
        # Training args
        training_args = TrainingArguments(
            output_dir='./bert_results',
            num_train_epochs=num_epochs,
            per_device_train_batch_size=16,
            warmup_steps=50,
            weight_decay=0.01,
            logging_dir='./bert_logs',
            logging_steps=50,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        trainer.train()
        
        return {
            'model': model,
            'tokenizer': tokenizer,
            'trainer': trainer
        }
        
    except ImportError:
        print("transformers library not installed. Install with: pip install transformers")
        return None


def create_zero_shot_prompt(text: str, categories: List[str]) -> str:
    """Create zero-shot classification prompt for LLM APIs."""
    prompt = f"""Classify the following government document into one of these categories:
{chr(10).join(f'- {cat}' for cat in categories)}

Document:
{text[:500]}

Respond with ONLY the category name, nothing else."""
    return prompt


if __name__ == '__main__':
    # Test classifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    texts = ["budget allocation", "policy implementation", "service request", "foia disclosure"]
    labels = [0, 1, 2, 3]
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    
    clf = TextClassifier('logistic_regression')
    clf.fit(X, labels)
    
    pred = clf.predict(X)
    print(f"Predictions: {pred}")
    print("Training module ready.")
