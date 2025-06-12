
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from preprocessing.preprocess_data import preprocess

np.random.seed(42)

# ---- 1. Import the data ----
feats = pd.read_csv('train_test_splits/train_split.feats.csv')
labels_0 = pd.read_csv('train_test_splits/train_split.labels.0.csv')

preprocess(feats)

print("Feats shape:", feats.shape)
print("Feats columns:", feats.columns)
print(feats.head())