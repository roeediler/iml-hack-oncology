
import numpy as np
import pandas as pd
import ast
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import ParameterSampler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, f1_score
from itertools import chain

sys.path.append(r"C:\\Users\\roeed\\OneDrive\\Documents\\Projects\\iml-hack-oncology")

from preprocessing.preprocess_data import preprocess
import preprocessing.data_completion as data_completion


np.random.seed(42)
top_k = 3

# ---- 1. Import the data ----
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

# ---- 3. Split the data into training and validation sets ----
X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_train, y_train, test_size=0.8, random_state=42)

# ---- 4. Hyperparameter tuning for KNN ----

# Define the distribution of hyperparameters for random search
param_dist = {
    'n_neighbors': np.arange(1, 51),  # Number of neighbors from 1 to 50
    'weights': ['uniform', 'distance'],  # Uniform or distance-weighted neighbors
    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']  # Distance metrics
}

# Set up the random search
random_search = RandomizedSearchCV(
    estimator=KNeighborsClassifier(),
    param_distributions=param_dist,
    n_iter=10,  # Number of parameter settings sampled
    cv=5,  # 5-fold cross-validation
    scoring= 'f1_macro',  # Use macro F1 score for evaluation
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Fit the random search to the training data
random_search.fit(X_train_val, y_train_val)

# Get the best model
best_knn_reg = random_search.best_estimator_
# Extract the best parameters from the best model
best_params = best_knn_reg.get_params()

# Create a descriptive sentence for each parameter
best_n_neighbors = best_params['n_neighbors']
best_weights = best_params['weights']
best_metric = best_params['metric']

# Print the descriptive sentences
print(f"The best model has {best_n_neighbors} nearest neighbors, uses '{best_weights}' weights, and the '{best_metric}' distance metric.")

# Print the performance of the best model
one_vs_rest_knn = OneVsRestClassifier(best_knn_reg)
one_vs_rest_knn.fit(X_train_val, y_train_val)
y_pred_knn = one_vs_rest_knn.predict(X_test_val)
print(f"prediction shape: {y_pred_knn[:5]}")
print("KNN Classification report:")
print(classification_report(y_test_val, y_pred_knn, target_names=mlb.classes_))

