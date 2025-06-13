# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LinearRegression, RidgeCV, Ridge
# from sklearn.model_selection import cross_val_score
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import VotingRegressor
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import mean_squared_error, make_scorer
# from preprocessing.preprocess_data import preprocess
# from preprocessing.data_completion import Clustering, Mean, DefaultValue
#
#
#
# def tumor_size_model(X_train,X_test,y_train):
#
#     # === 2. Define models ===
#     lin_model = LinearRegression()
#     ridge_model = RidgeCV(alphas=[0, 0.01, 0.1, 1, 10, 100, 1000], scoring='neg_mean_squared_error', cv=5)
#     knn_model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
#     tree_model = DecisionTreeRegressor(max_depth=4, random_state=0)
#
#     models = [lin_model, ridge_model, knn_model, tree_model]
#     names = ['linear', 'ridge', 'knn', 'tree']
#     mses = []
#
#     # === 3. Evaluate models with cross-validation on training set ===
#     for model in models:
#         neg_mse_scores = cross_val_score(
#             model, X_train, y_train,
#             scoring=make_scorer(mean_squared_error),
#             cv=5
#         )
#         avg_mse = np.mean(neg_mse_scores)
#         mses.append(avg_mse)
#
#     # === 4. Convert to weights using 1 / MSE
#     inverse_mses = [1 / m for m in mses]
#     total = sum(inverse_mses)
#     weights = [w / total for w in inverse_mses]  # normalize
#
#     print("✅ Computed weights (based only on X_train CV):")
#     for name, w, mse in zip(names, weights, mses):
#         print(f"{name:>6}: weight = {w:.3f}, avg CV MSE = {mse:.3f}")
#
#     # === 5. Refit models on all of X_train
#     for model in models:
#         model.fit(X_train, y_train)
#
#     # === 6. Build and evaluate ensemble on test set
#     ensemble = VotingRegressor(estimators=list(zip(names, models)), weights=weights)
#     ensemble.fit(X_train, y_train)
#     y_pred = ensemble.predict(X_test)
#     y_pred = np.round(y_pred, 1)
#     y_pred = np.maximum(y_pred, 0)
#
#     pd.Series(y_pred).to_csv("predictions.csv", index=False, header=False)
#
# if __name__ == "__main__":
#     data_complete = DefaultValue()
#
#     X_train = preprocess(pd.read_csv("../train_test_splits/train.feats.csv"), data_complete)
#     X_test = preprocess(pd.read_csv("../train_test_splits/test.feats.csv"), data_complete)
#     y_train = pd.read_csv("../train_test_splits/train.labels.1.csv").squeeze()
#
#     tumor_size_model(X_train,X_test,y_train)
#
#
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, make_scorer
from preprocessing.preprocess_data import preprocess
from preprocessing.data_completion import Clustering, Mean, DefaultValue



def tumor_size_model(X_train,X_test,y_train):

    # === 2. Define models ===
    lin_model = LinearRegression()
    ridge_model = RidgeCV(alphas=[0, 0.01, 0.1, 1, 10, 100, 1000], scoring='neg_mean_squared_error', cv=5)
    knn_model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5))
    tree_model = DecisionTreeRegressor(max_depth=4, random_state=0)

    models = [lin_model, ridge_model, knn_model, tree_model]
    names = ['linear', 'ridge', 'knn', 'tree']
    mses = []

    # === 3. Evaluate models with cross-validation on training set ===
    for model in models:
        neg_mse_scores = cross_val_score(
            model, X_train, y_train,
            scoring=make_scorer(mean_squared_error),
            cv=5
        )
        avg_mse = np.mean(neg_mse_scores)
        mses.append(avg_mse)

    # === 4. Convert to weights using 1 / MSE
    inverse_mses = [1 / m for m in mses]
    total = sum(inverse_mses)
    weights = [w / total for w in inverse_mses]  # normalize

    print("✅ Computed weights (based only on X_train CV):")
    for name, w, mse in zip(names, weights, mses):
        print(f"{name:>6}: weight = {w:.3f}, avg CV MSE = {mse:.3f}")

    # === 5. Refit models on all of X_train
    for model in models:
        model.fit(X_train, y_train)

    # === 6. Build and evaluate ensemble on test set
    ensemble = VotingRegressor(estimators=list(zip(names, models)), weights=weights)
    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    y_pred = np.round(y_pred, 1)
    y_pred = np.maximum(y_pred, 0)

    pd.Series(y_pred).to_csv("predictions.csv", index=False, header=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tumor size prediction using ensemble model.")

    parser.add_argument("train_feats", help="Path to training features CSV")
    parser.add_argument("train_labels", help="Path to training labels CSV")
    parser.add_argument("test_feats", help="Path to test features CSV")
    parser.add_argument("output", nargs="?", default="predictions.csv", help="Optional path to output predictions file")

    args = parser.parse_args()

    data_complete = DefaultValue()

    X_train = preprocess(pd.read_csv(args.train_feats), data_complete)
    X_test = preprocess(pd.read_csv(args.test_feats), data_complete)
    y_train = pd.read_csv(args.train_labels).squeeze()

    tumor_size_model(X_train,X_test,y_train,output_path=args.output)



