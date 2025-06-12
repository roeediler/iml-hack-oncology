import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
import os
father_folder = os.getcwd()

feats = pd.read_csv(f'{father_folder}/train_test_splits/train.feats.csv')
labels_0 = pd.read_csv(f'{father_folder}/train_test_splits/train.labels.0.csv')
labels_1 = pd.read_csv(f'{father_folder}/train_test_splits/train.labels.1.csv')

# Split the data
feats_train, feats_test, labels_0_train, labels_0_test, labels_1_train, labels_1_test = train_test_split(
    feats, labels_0, labels_1, test_size=0.2, random_state=42, shuffle=True
)

# Save the splits
feats_train.to_csv('train_test_splits/train_split.feats.csv', index=False)
feats_test.to_csv('train_test_splits/test_split.feats.csv', index=False)
labels_0_train.to_csv('train_test_splits/train_split.labels.0.csv', index=False)
labels_0_test.to_csv('train_test_splits/test_split.labels.0.csv', index=False)
labels_1_train.to_csv('train_test_splits/train_split.labels.1.csv', index=False)
labels_1_test.to_csv('train_test_splits/test_split.labels.1.csv', index=False)