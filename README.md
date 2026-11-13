# 🔍 FakeWatch — Fake News Detector

A one-page web app that classifies news articles as **REAL** or **FAKE** using NLP + Machine Learning.

---

## 📁 Project Structure

```
fake_news_detector/
├── app.py            ← Flask web server + one-page UI
├── train.py          ← Train the ML model
├── requirements.txt  ← Python dependencies
├── model.pkl         ← (generated after training)
├── vectorizer.pkl    ← (generated after training)
└── model_meta.json   ← (generated after training)
```

---

## 🚀 How to Run

### Step 1 — Install Python
Make sure Python 3.8+ is installed:
```bash
python --version
```

### Step 2 — Create Virtual Environment (recommended)
```bash
python -m venv venv

# Activate:
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Train the Model
```bash
python train.py
```
This will:
- Build a dataset of fake and real news samples
- Preprocess text (lowercase, remove URLs, punctuation)
- Extract TF-IDF features (up to 5000 tokens, bigrams)
- Train a Logistic Regression classifier
- Print accuracy, precision, recall, F1 score
- Save `model.pkl`, `vectorizer.pkl`, `model_meta.json`

### Step 5 — Start the Web App
```bash
python app.py
```

### Step 6 — Open in Browser
```
http://127.0.0.1:5000
```

---

## 🧪 How to Test

### In the Web UI:
1. Paste any news article into the text box
2. Click **"Analyse Article"**
3. See REAL / FAKE verdict + confidence score
4. Try the built-in sample buttons to test

### From the command line (curl):
```bash
curl -X POST http://127.0.0.1:5000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists discover miracle cure that doctors are hiding"}'
```

Expected response:
```json
{
  "label": "FAKE",
  "confidence": 87.3,
  "fake_prob": 87.3,
  "real_prob": 12.7
}
```

### Model status check:
```bash
curl http://127.0.0.1:5000/model_status
```

---

## 📚 What You'll Learn

| Concept | Where it appears |
|---|---|
| NLP Preprocessing | `train.py` → `preprocess()` function |
| TF-IDF Vectorization | `TfidfVectorizer` with bigrams |
| Text Classification | `LogisticRegression` model |
| Train/Test Split | `train_test_split()` with stratify |
| Accuracy | `accuracy_score()` |
| Precision | `precision_score()` |
| Recall | `recall_score()` |
| F1 Score | `f1_score()` |
| Confusion Matrix | `confusion_matrix()` |
| Model Persistence | `pickle.dump()` / `pickle.load()` |
| REST API | Flask `/classify` endpoint |

---

## 🔧 Upgrade Ideas

### Use a Real Dataset
Download from Kaggle:
- **WELFake Dataset** (72,000 articles)
- **ISOT Fake News Dataset**
- **FakeNewsNet**

Replace the sample arrays in `train.py` with:
```python
import pandas as pd
df = pd.read_csv('WELFake_Dataset.csv')
texts = df['text'].tolist()
labels = df['label'].tolist()  # 0=fake, 1=real
```

### Try Other Models
Replace `LogisticRegression` with:
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
```

### Try Deep Learning
```python
# pip install transformers torch
from transformers import pipeline
classifier = pipeline("text-classification", model="distilbert-base-uncased")
```

---

## ⚠️ Common Issues

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: flask` | Run `pip install -r requirements.txt` |
| `FileNotFoundError: model.pkl` | Run `python train.py` first |
| Port 5000 already in use | Change `port=5000` to `port=5001` in app.py |
| Low accuracy | Use a larger real dataset from Kaggle |
