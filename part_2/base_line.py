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
# """BASE LINE"""
#
# data_complete = Clustering()
# X_train = preprocess(pd.read_csv("../train_test_splits/train_split.feats.csv"))
# X_test = preprocess(pd.read_csv("../train_test_splits/test_split.feats.csv"))
# y_train = pd.read_csv("../train_test_splits/train_split.labels.1.csv").squeeze()
# y_test = pd.read_csv("../train_test_splits/test_split.labels.1.csv").squeeze()
#
#
# lin_model = LinearRegression()
# lin_model.fit(X_train, y_train)
# y_pred_lin = lin_model.predict(X_test)
# mse_lin = mean_squared_error(y_test, y_pred_lin)
#
#
# ridge_model = Ridge(alpha=10000000.0)
# ridge_model.fit(X_train, y_train)
# y_pred_ridge = ridge_model.predict(X_test)
# mse_ridge = mean_squared_error(y_test, y_pred_ridge)
#
#
# print(f"Linear Regression MSE: {mse_lin:.2f}")
# print(f"Ridge Regression (alpha=10000000) MSE: {mse_ridge:.2f}")
#
#
# pd.Series(y_pred_lin).to_csv("linear_predictions.csv", index=False, header=False)
# pd.Series(y_pred_ridge).to_csv("ridge_predictions.csv", index=False, header=False)

