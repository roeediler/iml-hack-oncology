import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, multilabel_confusion_matrix, hamming_loss, jaccard_score
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings('ignore')

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


def hyperparameter_tuning(X_tune, y_tune):
    """
    Perform hyperparameter tuning on 20% of the data
    """
    print("Starting hyperparameter tuning...")
    
    # Define parameter grid for AdaBoostClassifier
    param_grid = {
        'base_estimator__n_estimators': [50, 100, 200],
        'base_estimator__learning_rate': [0.1, 0.5, 1.0],
        'base_estimator__estimator__max_depth': [1, 2, 3]
    }
    
    # Create base estimator
    base_ada = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(),
        random_state=42
    )
    
    # Create ClassifierChain
    classifier_chain = ClassifierChain(base_ada, random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        classifier_chain,
        param_grid,
        cv=3,  # 3-fold cross-validation
        scoring='f1_macro',  # Good for multi-label classification
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_tune, y_tune)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_params_, grid_search.best_estimator_

def train_final_model(X_train, y_train, best_params):
    """
    Train the final model with best parameters on full dataset
    """
    print("\nTraining final model on full dataset...")
    
    # Create AdaBoost with best parameters
    best_ada = AdaBoostClassifier(
        n_estimators=best_params['base_estimator__n_estimators'],
        learning_rate=best_params['base_estimator__learning_rate'],
        base_estimator=DecisionTreeClassifier(
            max_depth=best_params['base_estimator__estimator__max_depth']
        ),
        random_state=42
    )
    
    # Create final ClassifierChain
    final_model = ClassifierChain(best_ada, random_state=42)
    
    # Train on full dataset
    final_model.fit(X_train, y_train)
    
    return final_model

def evaluate_model(model, X_test, y_test, target_cols):
    """
    Evaluate the trained model
    """
    print("\nEvaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    hamming = hamming_loss(y_test, y_pred)
    jaccard = jaccard_score(y_test, y_pred, average='macro')
    
    print(f"Hamming Loss: {hamming:.4f}")
    print(f"Jaccard Score (macro): {jaccard:.4f}")
    
    # Detailed classification report for each target
    print("\nDetailed Classification Report:")
    for i, col in enumerate(target_cols):
        print(f"\n--- {col} ---")
        print(classification_report(y_test.iloc[:, i], y_pred[:, i]))
    
    return y_pred

def main():
    """
    Main execution function
    """
        
    # Create sample data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 15
    n_targets = 4  # 4 different metastasis sites
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                    columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.DataFrame(np.random.randint(0, 2, (n_samples, n_targets)),
                    columns=['metastasis_liver', 'metastasis_lung', 
                            'metastasis_bone', 'metastasis_brain'])
    
    feature_cols = X.columns.tolist()
    target_cols = y.columns.tolist()
    
    print("Sample data created for demonstration")
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    # Split data: 20% for hyperparameter tuning, 80% for final training/testing
    X_tune, X_remaining, y_tune, y_remaining = train_test_split(
        X_scaled, y, test_size=0.8, random_state=42, stratify=y.iloc[:, 0]
    )
    
    # Further split remaining data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X_remaining, y_remaining, test_size=0.2, random_state=42, stratify=y_remaining.iloc[:, 0]
    )
    
    print(f"\nData splits:")
    print(f"Tuning set: {X_tune.shape[0]} samples")
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Hyperparameter tuning on 20% of data
    best_params, best_estimator = hyperparameter_tuning(X_tune, y_tune)
    
    # Train final model on full training data
    final_model = train_final_model(X_train, y_train, best_params)
    
    # Evaluate model
    y_pred = evaluate_model(final_model, X_test, y_test, target_cols)
    
    # Feature importance analysis
    print("\nFeature Importance Analysis:")
    # Get feature importance from the first classifier in the chain
    feature_importance = final_model.estimators_[0].feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(importance_df.head(10))
    
    print("\nModel training completed successfully!")
    
    return final_model, scaler, best_params

if __name__ == "__main__":
    model, scaler, params = main()
    
    # Save the model and scaler for later use
    import joblib
    joblib.dump(model, 'metastasis_classifier_chain.pkl')
    joblib.dump(scaler, 'feature_scaler.pkl')
    
    print("\nModel and scaler saved successfully!")
    print("Files saved: 'metastasis_classifier_chain.pkl', 'feature_scaler.pkl'")