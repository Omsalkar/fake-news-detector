"""
train.py — Train the Fake News Detection Model
Run with: python train.py

What this does:
1. Loads/generates dataset
2. Preprocesses text (NLP pipeline)
3. Extracts features with TF-IDF
4. Trains Logistic Regression classifier
5. Evaluates with accuracy, precision, recall, F1
6. Saves model + vectorizer for the Flask app
"""

import pickle
import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

# ─── SAMPLE DATASET ────────────────────────────────────────────────────────────
# In production: replace with WELFake / ISOT / Kaggle Fake News dataset
# Format: (text, label)  0 = FAKE, 1 = REAL

FAKE_SAMPLES = [
    "SHOCKING: Scientists discover that drinking bleach cures all diseases in 24 hours",
    "Government SECRETLY putting mind control chemicals in drinking water revealed",
    "BREAKING: Moon landing was staged, NASA insider finally admits the truth",
    "Big Pharma hiding miracle cancer cure made from common backyard plant",
    "5G towers are killing birds and causing autism in children proven",
    "Secret elite group controls all world governments and banks exposed",
    "Vaccines contain microchips to track your location says whistleblower",
    "MIRACLE: Woman grows back amputated leg using this one weird trick",
    "FBI insider leaks proof that elections have been rigged for decades",
    "URGENT: Share before deleted — dangerous truth about fluoride in water",
    "Hollywood actor reveals Illuminati plans to depopulate 90 percent of Earth",
    "Ancient alien technology found under pyramids proves human history is a lie",
    "Doctor fired for exposing truth that chemotherapy makes cancer worse not better",
    "Scientists bribed to hide evidence that Earth is actually flat not round",
    "New world order planning to implement one world government by next year",
    "Eating chocolate every day increases IQ by 40 points study proves",
    "Local man cures diabetes in 3 days using secret spice doctors hate",
    "Mainstream media caught fabricating entire Ukraine war crisis insider reveals",
    "Deep state planning to cancel elections and declare martial law sources say",
    "Prince reveals royal family are reptilian shapeshifters in explosive interview",
    "COVID vaccine makes you magnetic doctors confirm after viral video surfaces",
    "George Soros secretly funding antifa riots in every major American city",
    "Scientists confirm that sun is actually cold and NASA has been lying",
    "Man lives to 200 years old using ancient herb that pharma wants banned",
    "Climate change is a hoax invented by globalists to control energy sources",
    "Breaking exclusive: Barack Obama born in Kenya, new documents prove it",
    "PROOF: Sandy Hook was staged crisis actors hired paid by government",
    "Cell phones causing cancer epidemic and telecom companies paying off scientists",
    "Alternative cancer cure suppressed by medical establishment for decades revealed",
    "Shocking truth about aspartame it destroys brain cells over time confirmed",
    "Leaked memo shows CDC knew about vaccine dangers but hid it for years",
    "Alien spacecraft recovered in New Mexico according to declassified documents",
    "Child trafficking ring operating inside popular pizza restaurant exposed by anon",
    "Bill Gates planning to block the sun to cause global food shortage",
    "NASA faked Mars photos they were actually taken in desert in Arizona",
    "Doctors admit that aspirin causes more deaths than it prevents annually",
    "New study finds that WiFi radiation is deadlier than cigarettes long term",
    "Secret underground bunkers being built for elites before planned collapse",
    "Harvard professor fired for proving that gluten in food is mind control",
    "Confirmed: The earth is only 6000 years old and dinosaurs never existed",
]

REAL_SAMPLES = [
    "Federal Reserve holds interest rates steady citing economic uncertainty and inflation data",
    "NASA successfully launches Artemis mission with crew of four astronauts to lunar orbit",
    "Scientists publish new research on Alzheimer's treatment showing promising early results in trials",
    "Stock markets fall after disappointing jobs report shows slower than expected hiring growth",
    "United Nations Security Council meets to discuss escalating tensions in eastern Europe region",
    "Apple releases quarterly earnings beating analyst expectations with strong iPhone sales growth",
    "World Health Organization updates COVID booster recommendations based on new variant data",
    "Congress passes bipartisan infrastructure bill allocating funds for roads bridges and broadband",
    "Harvard researchers identify gene linked to increased risk of developing Type 2 diabetes",
    "European Central Bank raises interest rates by quarter point to combat persistent inflation",
    "Climate scientists warn Arctic sea ice at record low levels this summer season",
    "FDA approves new treatment for rare childhood leukemia showing 85 percent remission rate",
    "Tesla recalls 130000 vehicles over touchscreen overheating issue that affects autopilot display",
    "Pentagon releases new satellite imagery showing Russian troop movements near Ukrainian border",
    "Supreme Court agrees to hear case challenging state abortion restrictions next term",
    "California declares state of emergency as wildfires spread across northern counties this week",
    "Research shows Mediterranean diet significantly reduces risk of heart disease over decade",
    "New York City announces plan to reduce carbon emissions by 40 percent by 2030",
    "China launches crewed space mission to complete construction of Tiangong space station module",
    "World Bank warns developing nations face debt crisis as dollar interest rates rise",
    "Moderna announces new mRNA flu vaccine shows 80 percent efficacy in phase 3 trial",
    "Semiconductor shortage expected to ease by mid year as new fab plants come online",
    "Amazon reports $9.8 billion profit in third quarter driven by cloud computing growth",
    "Scientists discover new exoplanet in habitable zone with potential signs of water atmosphere",
    "OPEC plus agrees to cut oil production by 2 million barrels per day starting November",
    "Researchers develop biodegradable plastic alternative made from seaweed that dissolves in water",
    "Inflation drops to 3.2 percent in October marking sixth consecutive month of decline",
    "EU announces new regulations requiring tech companies to give users data portability rights",
    "Johnson and Johnson agrees to 8.9 billion settlement over talc baby powder cancer claims",
    "MIT team develops new battery technology that charges to 80 percent capacity in 5 minutes",
    "Ukraine receives additional military aid package from US including air defense systems",
    "Economists predict moderate recession risk in 2024 as consumer spending shows signs of slowing",
    "New archaeological discovery in Turkey pushes back evidence of human settlement by 3000 years",
    "Department of Justice files antitrust lawsuit against major airline merger over competition concerns",
    "WHO declares end to mpox public health emergency after cases decline globally over months",
    "Federal budget deficit narrows to 1.7 trillion as tax revenues increase and spending slows",
    "SpaceX successfully lands Starship prototype completing first full flight test without explosion",
    "Scientists use CRISPR gene editing to successfully treat sickle cell disease in clinical trial",
    "Bank of England raises rates to highest level in 15 years to fight stubborn inflation",
    "Global average temperature in 2023 confirmed as hottest year on record by multiple agencies",
]

# Extra augmented samples for better generalization
EXTRA_FAKE = [
    "Actor reveals all celebrities worship Satan at secret underground rituals annually",
    "BOMBSHELL: Mainstream scientist admits global warming data has been manipulated for funding",
    "Hidden cure for baldness discovered that hair loss industry has suppressed for 50 years",
    "Secret treaty gives China control of US military bases by 2025 leaked documents show",
    "Drinking lemon water every morning reverses cancer studies find but doctors deny it",
]

EXTRA_REAL = [
    "Britain's economy grew by 0.2 percent in the third quarter beating analyst forecasts",
    "Pentagon confirms it will provide Ukraine with additional Patriot air defense batteries",
    "Scientists identify protein that plays key role in spreading of pancreatic cancer cells",
    "Consumer confidence index falls to six month low amid concerns about household debt",
    "International Monetary Fund upgrades global growth forecast to 3.1 percent for current year",
]

# ─── BUILD DATASET ─────────────────────────────────────────────────────────────
texts  = FAKE_SAMPLES + EXTRA_FAKE + REAL_SAMPLES + EXTRA_REAL
labels = ([0] * (len(FAKE_SAMPLES) + len(EXTRA_FAKE))) + \
         ([1] * (len(REAL_SAMPLES)  + len(EXTRA_REAL)))


# ─── NLP PREPROCESSING ─────────────────────────────────────────────────────────
def preprocess(text):
    """Clean and normalize text."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)        # remove URLs
    text = re.sub(r'[^a-z\s]', '', text)               # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()           # normalize whitespace
    return text

print("\n" + "="*50)
print("  FAKE NEWS DETECTOR — MODEL TRAINING")
print("="*50)
print(f"\n📚 Dataset: {len(texts)} samples")
print(f"   • Fake: {labels.count(0)}")
print(f"   • Real: {labels.count(1)}")

# Preprocess all texts
texts_clean = [preprocess(t) for t in texts]

# ─── TRAIN / TEST SPLIT ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    texts_clean, labels, test_size=0.2, random_state=42, stratify=labels
)
print(f"\n📊 Split: {len(X_train)} train / {len(X_test)} test samples")

# ─── FEATURE EXTRACTION (TF-IDF) ──────────────────────────────────────────────
print("\n🔤 Extracting TF-IDF features...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),      # unigrams + bigrams
    sublinear_tf=True,        # log normalization
    min_df=1,
    stop_words='english'
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)
print(f"   • Vocabulary size: {len(vectorizer.vocabulary_)} tokens")

# ─── MODEL TRAINING ────────────────────────────────────────────────────────────
print("\n🤖 Training Logistic Regression classifier...")
model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver='lbfgs',
    class_weight='balanced'
)
model.fit(X_train_vec, y_train)

# ─── EVALUATION ────────────────────────────────────────────────────────────────
y_pred       = model.predict(X_test_vec)
y_train_pred = model.predict(X_train_vec)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc  = accuracy_score(y_test,  y_pred)
prec      = precision_score(y_test, y_pred, zero_division=0)
rec       = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)

print("\n" + "─"*50)
print("  📈 EVALUATION RESULTS")
print("─"*50)
print(f"  Training Accuracy : {train_acc:.1%}")
print(f"  Test Accuracy     : {test_acc:.1%}")
print(f"  Precision         : {prec:.1%}  (of predicted REAL, how many truly REAL)")
print(f"  Recall            : {rec:.1%}  (of all REAL, how many found)")
print(f"  F1 Score          : {f1:.1%}  (harmonic mean of P & R)")
print("─"*50)

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["FAKE", "REAL"]))

cm = confusion_matrix(y_test, y_pred)
print("  Confusion Matrix:")
print(f"              Predicted FAKE  Predicted REAL")
print(f"  Actual FAKE      {cm[0][0]:>4}            {cm[0][1]:>4}")
print(f"  Actual REAL      {cm[1][0]:>4}            {cm[1][1]:>4}")
print()

# ─── SAVE MODEL ────────────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

meta = {
    "model_type": "Logistic Regression + TF-IDF",
    "train_accuracy": f"{train_acc:.1%}",
    "test_accuracy":  f"{test_acc:.1%}",
    "precision":      f"{prec:.1%}",
    "recall":         f"{rec:.1%}",
    "f1":             f"{f1:.1%}",
    "vocab_size":     len(vectorizer.vocabulary_),
    "train_samples":  len(X_train),
    "test_samples":   len(X_test),
}
with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("✅ Saved: model.pkl, vectorizer.pkl, model_meta.json")
print("\n🚀 Now run: python app.py")
print("   Open:    http://127.0.0.1:5000\n")
