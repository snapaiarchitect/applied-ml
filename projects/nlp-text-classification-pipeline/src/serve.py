"""
FastAPI serving module for NLP text classification models.
Production-ready API for real-time document classification.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import pickle
import os

app = FastAPI(title="Document Classification API", version="1.0.0")

# Global model storage (load on startup)
models = {}
vectorizers = {}


class DocumentRequest(BaseModel):
    text: str
    model_type: Optional[str] = "logistic_regression"


class BatchDocumentRequest(BaseModel):
    documents: List[str]
    model_type: Optional[str] = "logistic_regression"


class ClassificationResponse(BaseModel):
    text: str
    predicted_category: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None


class BatchClassificationResponse(BaseModel):
    results: List[ClassificationResponse]


def load_models(model_dir: str = '../models'):
    """Load all available models on startup."""
    global models, vectorizers
    
    # Load logistic regression model
    lr_path = os.path.join(model_dir, 'logistic_regression.pkl')
    if os.path.exists(lr_path):
        with open(lr_path, 'rb') as f:
            data = pickle.load(f)
        models['logistic_regression'] = data['model']
        print("Loaded logistic regression model")
    
    # Load vectorizer
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    if os.path.exists(vectorizer_path):
        with open(vectorizer_path, 'rb') as f:
            vectorizers['tfidf'] = pickle.load(f)
        print("Loaded TF-IDF vectorizer")
    
    # Load BERT if available
    bert_path = os.path.join(model_dir, 'bert_document_classifier')
    if os.path.exists(bert_path):
        try:
            from transformers import BertForSequenceClassification, BertTokenizer
            models['bert'] = BertForSequenceClassification.from_pretrained(bert_path)
            vectorizers['bert'] = BertTokenizer.from_pretrained(bert_path)
            print("Loaded BERT model")
        except ImportError:
            print("transformers not installed, skipping BERT")


@app.on_event("startup")
async def startup_event():
    """Load models on API startup."""
    load_models()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "version": "1.0.0"
    }


@app.post("/classify", response_model=ClassificationResponse)
async def classify_document(request: DocumentRequest):
    """Classify a single document."""
    if request.model_type not in models:
        raise HTTPException(status_code=400, detail=f"Model {request.model_type} not loaded")
    
    model = models[request.model_type]
    
    if request.model_type == 'logistic_regression':
        # Preprocess and vectorize
        from .preprocess import preprocess_text
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        cleaned_text = preprocess_text(request.text)
        
        if 'tfidf' not in vectorizers:
            raise HTTPException(status_code=500, detail="Vectorizer not loaded")
        
        vectorizer = vectorizers['tfidf']
        X = vectorizer.transform([cleaned_text])
        
        # Predict
        pred = model.predict(X)[0]
        proba = model.predict_proba(X)[0]
        
        # Map to category name
        label_map = {0: 'Budget', 1: 'Policy', 2: 'Service_Request', 3: 'FOIA'}
        category = label_map.get(pred, str(pred))
        
        return ClassificationResponse(
            text=request.text[:100],
            predicted_category=category,
            confidence=float(max(proba)),
            probabilities={label_map.get(i, str(i)): float(p) for i, p in enumerate(proba)}
        )
    
    else:
        raise HTTPException(status_code=400, detail="Only logistic_regression supported currently")


@app.post("/classify/batch", response_model=BatchClassificationResponse)
async def classify_documents(request: BatchDocumentRequest):
    """Classify multiple documents."""
    results = []
    
    for text in request.documents:
        doc_request = DocumentRequest(text=text, model_type=request.model_type)
        result = await classify_document(doc_request)
        results.append(result)
    
    return BatchClassificationResponse(results=results)


if __name__ == '__main__':
    # Test the API
    import uvicorn
    
    print("Starting Document Classification API...")
    print("Test with: curl -X POST http://localhost:8000/classify -H 'Content-Type: application/json' -d '{\"text\": \"budget allocation document\"}'")
    
    # Run with: uvicorn serve:app --reload
    # For testing without running server:
    print("\nAPI module ready. Run with: uvicorn src.serve:app --host 0.0.0.0 --port 8000")
