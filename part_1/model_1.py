
import numpy as np
import pandas as pd
import ast
import sys

from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score

from itertools import chain

sys.path.append(r"C:\\Users\\roeed\\OneDrive\\Documents\\Projects\\iml-hack-oncology")

from preprocessing.preprocess_data import preprocess
import preprocessing.data_completion as data_completion

import os

np.random.seed(42)
top_k = 3

# ---- 1. Import the data ----
father_folder = os.path.dirname(os.getcwd())
X = pd.read_csv('train_test_splits/train_split.feats.csv')
y_raw = pd.read_csv('train_test_splits/train_split.labels.0.csv')

# change the column name to 'metastasis' for consistency
y_raw.rename(columns={'אבחנה-Location of distal metastases': 'metastasis'}, inplace=True)

# Convert string to actual list
y_raw['metastasis'] = y_raw['metastasis'].apply(ast.literal_eval)

preprocess(X)

# Convert the 'metastasis' column to a list of unique labels
y_raw = y_raw['metastasis'].tolist()

# convert inner empty lists to list with the string 'None'
y_raw = [labels if labels else ['None'] for labels in y_raw]

possible_labels = list(set(chain.from_iterable(y_raw)))

# Binarize labels
mlb = MultiLabelBinarizer(classes=possible_labels)
y = mlb.fit_transform(y_raw)

# ---- 2. Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Using a Random Forest classifier with OneVsRest strategy
model = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
model.fit(X_train, y_train)

# ---- 4. Prediction with top-3 filtering ----
y_proba = model.predict_proba(X_test)

# Apply top-3 threshold per patient
y_pred_topk = np.zeros_like(y_proba)
for i in range(y_proba.shape[0]):
    top_indices = np.argsort(y_proba[i])[-top_k:]
    y_pred_topk[i, top_indices] = 1


# ---- 5. Evaluation ----
print("Classification report with top-3 filtered predictions:")
print(classification_report(y_test, y_pred_topk, target_names=mlb.classes_))
print("Micro-F1:", f1_score(y_test, y_pred_topk, average='micro'))
print("Macro-F1:", f1_score(y_test, y_pred_topk, average='macro'))


# --- 6. Randomized Search for Hyperparameter Tuning ---
# Define hyperparameter space for AdaBoost
param_dist_ada = {
    'n_estimators': np.arange(25, 401, 25),
    'learning_rate': np.linspace(0.01, 2.0, 20),
    'estimator': [
        DecisionTreeClassifier(max_depth=d) for d in range(1, 6)
    ], 
}

# Set up random search for AdaBoost
random_search_ada = ClassifierChain(RandomizedSearchCV(
    estimator=AdaBoostClassifier(random_state=42),
    param_distributions=param_dist_ada,
    n_iter=20,
    cv=5,
    scoring='f1_macro',
    verbose=1,
    n_jobs=-1,
    random_state=42
), order='random', random_state=42)

random_search_ada.fit(X_train, y_train)

# Predict on test/validation split
y_pred_ada = random_search_ada.predict(X_test)

# Print classification report
print("AdaBoost Classification report:")
print(classification_report(y_test, y_pred_ada, target_names=mlb.classes_))