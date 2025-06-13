
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

from preprocessing.preprocess_data import preprocess
import preprocessing.data_completion as data_completion
import os

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict metastases locations")
    parser.add_argument("--train_samples", required=True,
                        help="Path to the training samples csv file")
    parser.add_argument("--train_labels", required=True,
                        help="Path to the training labels csv file")
    parser.add_argument("--test_samples", required=True,
                        help="Path to the test samples csv file")
    parser.add_argument("--output", default="predictions.csv",
                        help="Path to the output file (were the predicted "
                             "labels for the test samples will be saved")

    args = parser.parse_args()

    np.random.seed(42)
    top_k = 3

    # ---- 1. Import the data ----
    # X_full = pd.read_csv('train_test_splits/train.feats.csv')
    # preprocess(X_full)
    # print("Full training features shape:", X_full.shape)
    # Load the training features and labels
    X = pd.read_csv(args.train_samples)
    y_raw = pd.read_csv(args.train_labels)

    # Load the test features for the final predictions
    X_test_feats = pd.read_csv(args.test_samples)
    preprocess(X_test_feats)

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
    # print("Classification report with top-3 filtered predictions:")
    # print(classification_report(y_test, y_pred_topk, target_names=mlb.classes_))
    # print("Micro-F1:", f1_score(y_test, y_pred_topk, average='micro'))
    # print("Macro-F1:", f1_score(y_test, y_pred_topk, average='macro'))


    # --- 6. Randomized Search for Hyperparameter Tuning ---

    # Split the training data into a smaller training and validation set for hyperparameter tuning
    X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_train, y_train, test_size=0.8, random_state=42)

    # Define parameter grid for AdaBoostClassifier
    param_dist_ada = {
        'estimator__n_estimators': np.arange(25, 201, 25),
        'estimator__learning_rate': np.linspace(0.01, 2.0, 20),
        'estimator__estimator__max_depth': [1, 2, 3, 4, 5, 6]
    }

    # Create base estimator
    base_ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(),
        random_state=42
    )

    # Create ClassifierChain
    classifier_chain = ClassifierChain(base_ada, random_state=42)

    # Set up random search for AdaBoost
    random_search_ada = RandomizedSearchCV(
        estimator=classifier_chain,
        param_distributions=param_dist_ada,
        n_iter=20,
        cv=5,
        scoring='f1_macro',
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    # Fit the model
    random_search_ada.fit(X_train_val, y_train_val)

    # Train the final model with best parameters on the full training set
    best_params_ada = random_search_ada.best_params_
    # print(f"Best parameters for AdaBoost: {best_params_ada}")
    # print(f"Best cross-validation score for AdaBoost: {random_search_ada.best_score_:.4f}")
    # Train the final model with best parameters on full training set
    best_ada = AdaBoostClassifier(
        n_estimators=best_params_ada['estimator__n_estimators'],
        learning_rate=best_params_ada['estimator__learning_rate'],
        estimator=DecisionTreeClassifier(
            max_depth=best_params_ada['estimator__estimator__max_depth']
        ),
        random_state=42
    )

    classifier_chain_ada = ClassifierChain(best_ada, random_state=42)

    # Fit the final model on the full training set
    classifier_chain_ada.fit(X_train, y_train)

    # Predict on test/validation split
    y_pred_ada = classifier_chain_ada.predict(X_test)

    # Print classification report
    # print("AdaBoost Classification report:")
    # print(classification_report(y_test, y_pred_ada, target_names=mlb.classes_))

    # Convert prediction vectors to lists of metastasis sites using mlb.classes_
    def vectors_to_metastasis_lists(y_pred, mlb):
        preds = [
            [mlb.classes_[i] for i, val in enumerate(row) if int(val) == 1]
            for row in y_pred
        ]
        # remove 'None' from the lists
        preds = [[label for label in labels if label != 'None'] for labels in preds]
        # convert the inner lists to strings
        preds = [str(labels) for labels in preds]
        preds = pd.DataFrame(preds, columns=['אבחנה-Location of distal metastases'])
        return preds

    # Make predictions on the test set
    y_preds = classifier_chain_ada.predict(X_test_feats)
    y_preds = vectors_to_metastasis_lists(y_preds, mlb)
    # Save predictions to CSV
    y_preds.to_csv(args.output, index=False, encoding='utf-8-sig', header=True)


    # Make predictions on the full training set to ensure consistency
    # y_full_preds = classifier_chain_ada.predict(X_full)
    # y_full_preds = vectors_to_metastasis_lists(y_full_preds, mlb)
    # # Save full predictions to CSV
    # y_full_preds.to_csv('train_test_splits/train_predictions_metastasis.csv', index=False, encoding='utf-8-sig', header=True)


#
# import numpy as np
# import pandas as pd
# import ast
# import sys
#
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.multioutput import ClassifierChain
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
#
# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.multiclass import OneVsRestClassifier
# from sklearn.model_selection import train_test_split, RandomizedSearchCV
# from sklearn.metrics import classification_report, f1_score
#
# from itertools import chain
#
# sys.path.append(r"C:\\Users\\roeed\\OneDrive\\Documents\\Projects\\iml-hack-oncology")
#
# from preprocessing.preprocess_data import preprocess
# import preprocessing.data_completion as data_completion
#
# import os
#
# np.random.seed(42)
# top_k = 3
#
# # ---- 1. Import the data ----
# X_full = pd.read_csv('train_test_splits/train.feats.csv')
# preprocess(X_full)
# print("Full training features shape:", X_full.shape)
# # Load the training features and labels
# X = pd.read_csv('train_test_splits/train_split.feats.csv')
# y_raw = pd.read_csv('train_test_splits/train_split.labels.0.csv')
#
# # Load the test features for the final predictions
# X_test_feats = pd.read_csv('train_test_splits/test.feats.csv')
# preprocess(X_test_feats)
#
# # change the column name to 'metastasis' for consistency
# y_raw.rename(columns={'אבחנה-Location of distal metastases': 'metastasis'}, inplace=True)
#
# # Convert string to actual list
# y_raw['metastasis'] = y_raw['metastasis'].apply(ast.literal_eval)
#
# preprocess(X)
#
# # Convert the 'metastasis' column to a list of unique labels
# y_raw = y_raw['metastasis'].tolist()
#
# # convert inner empty lists to list with the string 'None'
# y_raw = [labels if labels else ['None'] for labels in y_raw]
#
# possible_labels = list(set(chain.from_iterable(y_raw)))
#
# # Binarize labels
# mlb = MultiLabelBinarizer(classes=possible_labels)
# y = mlb.fit_transform(y_raw)
#
# # ---- 2. Train/test split ----
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Using a Random Forest classifier with OneVsRest strategy
# model = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
# model.fit(X_train, y_train)
#
# # ---- 4. Prediction with top-3 filtering ----
# y_proba = model.predict_proba(X_test)
#
# # Apply top-3 threshold per patient
# y_pred_topk = np.zeros_like(y_proba)
# for i in range(y_proba.shape[0]):
#     top_indices = np.argsort(y_proba[i])[-top_k:]
#     y_pred_topk[i, top_indices] = 1
#
#
# # ---- 5. Evaluation ----
# print("Classification report with top-3 filtered predictions:")
# print(classification_report(y_test, y_pred_topk, target_names=mlb.classes_))
# print("Micro-F1:", f1_score(y_test, y_pred_topk, average='micro'))
# print("Macro-F1:", f1_score(y_test, y_pred_topk, average='macro'))
#
#
# # --- 6. Randomized Search for Hyperparameter Tuning ---
#
# # Split the training data into a smaller training and validation set for hyperparameter tuning
# X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_train, y_train, test_size=0.8, random_state=42)
#
# # Define parameter grid for AdaBoostClassifier
# param_dist_ada = {
#     'base_estimator__n_estimators': np.arange(25, 201, 25),
#     'base_estimator__learning_rate': np.linspace(0.01, 2.0, 20),
#     'base_estimator__estimator__max_depth': [1, 2, 3, 4, 5, 6]
# }
#
# # Create base estimator
# base_ada = AdaBoostClassifier(
#     estimator=DecisionTreeClassifier(),
#     random_state=42
# )
#
# # Create ClassifierChain
# classifier_chain = ClassifierChain(base_ada, random_state=42)
#
# # Set up random search for AdaBoost
# random_search_ada = RandomizedSearchCV(
#     estimator=classifier_chain,
#     param_distributions=param_dist_ada,
#     n_iter=20,
#     cv=5,
#     scoring='f1_macro',
#     verbose=1,
#     n_jobs=-1,
#     random_state=42
# )
#
# # Fit the model
# random_search_ada.fit(X_train_val, y_train_val)
#
# # Train the final model with best parameters on the full training set
# best_params_ada = random_search_ada.best_params_
# print(f"Best parameters for AdaBoost: {best_params_ada}")
# print(f"Best cross-validation score for AdaBoost: {random_search_ada.best_score_:.4f}")
# # Train the final model with best parameters on full training set
# best_ada = AdaBoostClassifier(
#     n_estimators=best_params_ada['base_estimator__n_estimators'],
#     learning_rate=best_params_ada['base_estimator__learning_rate'],
#     estimator=DecisionTreeClassifier(
#         max_depth=best_params_ada['base_estimator__estimator__max_depth']
#     ),
#     random_state=42
# )
#
# classifier_chain_ada = ClassifierChain(best_ada, random_state=42)
#
# # Fit the final model on the full training set
# classifier_chain_ada.fit(X_train, y_train)
#
# # Predict on test/validation split
# y_pred_ada = classifier_chain_ada.predict(X_test)
#
# # Print classification report
# print("AdaBoost Classification report:")
# print(classification_report(y_test, y_pred_ada, target_names=mlb.classes_))
#
# # Convert prediction vectors to lists of metastasis sites using mlb.classes_
# def vectors_to_metastasis_lists(y_pred, mlb):
#     preds = [
#         [mlb.classes_[i] for i, val in enumerate(row) if int(val) == 1]
#         for row in y_pred
#     ]
#     # remove 'None' from the lists
#     preds = [[label for label in labels if label != 'None'] for labels in preds]
#     # convert the inner lists to strings
#     preds = [str(labels) for labels in preds]
#     preds = pd.DataFrame(preds, columns=['אבחנה-Location of distal metastases'])
#     return preds
#
# # Make predictions on the test set
# y_preds = classifier_chain_ada.predict(X_test_feats)
# y_preds = vectors_to_metastasis_lists(y_preds, mlb)
# # Save predictions to CSV
# y_preds.to_csv('train_test_splits/predictions_metastases.csv', index=False, encoding='utf-8-sig', header=True)
#
#
#
# # Make predictions on the full training set to ensure consistency
# y_full_preds = classifier_chain_ada.predict(X_full)
# y_full_preds = vectors_to_metastasis_lists(y_full_preds, mlb)
# # Save full predictions to CSV
# y_full_preds.to_csv('train_test_splits/train_predictions_metastasis.csv', index=False, encoding='utf-8-sig', header=True)







