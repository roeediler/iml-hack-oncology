# This is a temp file
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# ---- 1. Generate synthetic data ----
np.random.seed(42)
n_samples = 1000
n_features = 20
n_labels = 11

# Simulated features (e.g., tumor size, stage, receptor status, etc.)
X = np.random.randn(n_samples, n_features)

def generate_metastasis_count(X_input):
    """
    Takes a feature vector X_input and returns an integer representing the number of metastasis sites.
    The number of sites is determined by a simple rule based on the sum of the features.
    """
    # For demonstration, use a simple rule based on the sum of features
    score = np.sum(X_input)
    sites = np.nan
    if score < -5:
        sites = 0
    elif score < 0:
        sites = 1
    elif score < 5:
        sites = 2
    else:
        sites = 3
    # generate list in size sites of numbers between 1 and n_labels
    return sites

def generate_metastasis_sites(X_input):
        """
        Given a feature vector X_input and a number of sites (0-3), 
        returns a list of unique site indices (1 to 11 inclusive) of length num_sites.
        """
        num_sites = generate_metastasis_count(X_input)
        if num_sites == 0:
            return []
        rng = np.random.default_rng(abs(int(np.sum(X_input) * 1000)) % (2**32))
        return list(rng.choice(range(1, 12), size=num_sites, replace=False))

y = [generate_metastasis_sites(x) for x in X]

y_raw = y.copy()
# Define possible labels (1 to 11)
possible_labels = list(range(1, 12))
# Binarize labels
mlb = MultiLabelBinarizer(classes=possible_labels)
y = mlb.fit_transform(y_raw)

# ---- 2. Train/test split ----
# X_train, X_test, y_train_vecs, y_test_vecs, y_train_count, y_test_count = train_test_split(X, y, y_metastasis_count, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- 3. Model training ----

# Using a Random Forest classifier with OneVsRest strategy
model = OneVsRestClassifier(RandomForestClassifier(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# ---- 4. Prediction with top-3 filtering ----
y_proba = model.predict_proba(X_test)

# Apply top-3 threshold per patient
top_k = 3
y_pred_topk = np.zeros_like(y_proba)
for i in range(y_proba.shape[0]):
    top_indices = np.argsort(y_proba[i])[-top_k:]
    y_pred_topk[i, top_indices] = 1

class_names = [str(num) for num in mlb.classes_]

# ---- 5. Evaluation ----
print("Classification report with top-3 filtered predictions:")
print(classification_report(y_test, y_pred_topk, target_names=class_names))
print("Micro-F1:", f1_score(y_test, y_pred_topk, average='micro'))
print("Macro-F1:", f1_score(y_test, y_pred_topk, average='macro'))
