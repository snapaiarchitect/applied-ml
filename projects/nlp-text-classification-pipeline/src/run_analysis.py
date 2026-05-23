"""
NLP Text Classification Analysis — 20 Newsgroups (REAL DATA)
Lightweight version: fixed hyperparameters, fast execution, all real.
"""
import json
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix
)
from nltk.corpus import stopwords
import nltk

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

STOPWORDS = set(stopwords.words('english'))

PROJECT = Path(__file__).resolve().parents[1]
FIGURES = PROJECT / 'figures'
FIGURES.mkdir(exist_ok=True)

# ============================================================
# 1. Load REAL 20 Newsgroups
# ============================================================
print("Loading REAL 20 Newsgroups data...")
train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'), shuffle=True, random_state=42)
print(f"Train: {len(train.data):,} | Test: {len(test.data):,} | Classes: {len(train.target_names)}")

# ============================================================
# 2. Preprocessing
# ============================================================
print("Preprocessing...")

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return ' '.join(tokens)

X_train = [preprocess(t) for t in train.data]
X_test = [preprocess(t) for t in test.data]

# ============================================================
# 3. TF-IDF
# ============================================================
print("TF-IDF vectorization...")
vec = TfidfVectorizer(max_features=15000, min_df=2, max_df=0.95, sublinear_tf=True)
X_train_tfidf = vec.fit_transform(X_train)
X_test_tfidf = vec.transform(X_test)
print(f"Vocab: {len(vec.vocabulary_):,} | Shape: {X_train_tfidf.shape}")

# ============================================================
# 4. Class Distribution
# ============================================================
print("Class distribution chart...")
train_counts = pd.Series(train.target).value_counts().sort_index()
train_counts.index = [train.target_names[i] for i in train_counts.index]

plt.figure(figsize=(12, 5))
sns.barplot(x=train_counts.index, y=train_counts.values, hue=train_counts.index, palette='viridis', legend=False)
plt.xticks(rotation=45, ha='right')
plt.title('20 Newsgroups — Class Distribution (Training Set)')
plt.ylabel('Documents')
plt.tight_layout()
plt.savefig(FIGURES / 'class_distribution.png', dpi=150)
plt.close()

# ============================================================
# 5. Train Models (fixed hyperparameters for speed)
# ============================================================
print("Training models...")
results = {}

# Naive Bayes
nb = MultinomialNB(alpha=0.1)
nb.fit(X_train_tfidf, train.target)
nb_pred = nb.predict(X_test_tfidf)
nb_acc = accuracy_score(test.target, nb_pred)
results['Naive Bayes'] = {'accuracy': nb_acc, 'predictions': nb_pred}
print(f"  Naive Bayes:     {nb_acc:.4f}")

# Logistic Regression
lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
lr.fit(X_train_tfidf, train.target)
lr_pred = lr.predict(X_test_tfidf)
lr_acc = accuracy_score(test.target, lr_pred)
results['Logistic Regression'] = {'accuracy': lr_acc, 'predictions': lr_pred}
print(f"  Logistic Reg:    {lr_acc:.4f}")

# Linear SVM
svm = LinearSVC(C=1.0, max_iter=3000, random_state=42)
svm.fit(X_train_tfidf, train.target)
svm_pred = svm.predict(X_test_tfidf)
svm_acc = accuracy_score(test.target, svm_pred)
results['Linear SVM'] = {'accuracy': svm_acc, 'predictions': svm_pred}
print(f"  Linear SVM:      {svm_acc:.4f}")

# ============================================================
# 6. Model Comparison
# ============================================================
print("Model comparison chart...")
names = list(results.keys())
accs = [results[n]['accuracy'] for n in names]

plt.figure(figsize=(7, 5))
sns.barplot(x=names, y=accs, hue=names, palette='muted', legend=False)
plt.ylim(0, 1)
plt.title('Model Accuracy Comparison (20 Newsgroups Test Set)')
plt.ylabel('Accuracy')
for i, v in enumerate(accs):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES / 'model_comparison.png', dpi=150)
plt.close()

# ============================================================
# 7. Confusion Matrix (best model)
# ============================================================
print("Confusion matrix...")
best = max(results, key=lambda x: results[x]['accuracy'])
best_pred = results[best]['predictions']
cm = confusion_matrix(test.target, best_pred)

plt.figure(figsize=(13, 11))
sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=train.target_names, yticklabels=train.target_names)
plt.title(f'Confusion Matrix — {best} (20 Newsgroups)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(FIGURES / 'confusion_matrix.png', dpi=150)
plt.close()

# ============================================================
# 8. Per-Class Metrics
# ============================================================
print("Per-class metrics...")
for name in names:
    pred = results[name]['predictions']
    p, r, f1, _ = precision_recall_fscore_support(test.target, pred, average='weighted')
    print(f"  {name:20s} | P={p:.4f} R={r:.4f} F1={f1:.4f}")

# ============================================================
# 9. Top TF-IDF Features per Class
# ============================================================
print("Top TF-IDF features chart...")
feat_names = vec.get_feature_names_out()
coef = lr.coef_

fig, axes = plt.subplots(5, 4, figsize=(16, 20))
axes = axes.flatten()
for i, cls in enumerate(train.target_names):
    top_idx = np.argsort(coef[i])[-10:][::-1]
    words = [feat_names[j] for j in top_idx]
    scores = np.sort(coef[i])[-10:][::-1]
    ax = axes[i]
    sns.barplot(x=scores, y=words, hue=words, palette='rocket', legend=False, ax=ax)
    ax.set_title(cls, fontsize=9)
    ax.set_xlabel('Weight')
plt.suptitle('Top 10 TF-IDF Features per Newsgroup (Logistic Regression)', fontsize=14, y=1.005)
plt.tight_layout()
plt.savefig(FIGURES / 'top_tfidf_features.png', dpi=150)
plt.close()

# ============================================================
# 10. Save Results
# ============================================================
report = {
    "dataset": "sklearn.datasets.fetch_20newsgroups",
    "source_note": "Derived from CMU 20 Newsgroups (Lang, 1995). ~18K Usenet posts across 20 topics.",
    "train_samples": len(train.data),
    "test_samples": len(test.data),
    "n_classes": len(train.target_names),
    "vocab_size": len(vec.vocabulary_),
    "models": {n: {"accuracy": round(results[n]['accuracy'], 4)} for n in names},
    "best_model": best,
    "figures": ["class_distribution.png", "model_comparison.png", "confusion_matrix.png", "top_tfidf_features.png"]
}
with open(PROJECT / 'results.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 60)
print("COMPLETE — All outputs from REAL 20 Newsgroups data")
print("=" * 60)
print(json.dumps(report, indent=2))
