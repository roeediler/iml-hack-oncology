# import pandas as pd
# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.metrics import mean_squared_error
#
# X_train = pd.read_csv("X_train.csv")
# X_test = pd.read_csv("X_test.csv")
# y_train = pd.read_csv("y_train.csv").squeeze()
# y_test = pd.read_csv("y_test.csv").squeeze()
#
#
# lin_model = LinearRegression()
# lin_model.fit(X_train, y_train)
# y_pred_lin = lin_model.predict(X_test)
# mse_lin = mean_squared_error(y_test, y_pred_lin)
#
#
# ridge_model = Ridge(alpha=1000.0)
# ridge_model.fit(X_train, y_train)
# y_pred_ridge = ridge_model.predict(X_test)
# mse_ridge = mean_squared_error(y_test, y_pred_ridge)
#
#
# print(f"Linear Regression MSE: {mse_lin:.2f}")
# print(f"Ridge Regression (alpha=1000) MSE: {mse_ridge:.2f}")
#
# #
# #pd.Series(y_pred_lin).to_csv("linear_predictions.csv", index=False, header=False)
# #pd.Series(y_pred_ridge).to_csv("ridge_predictions.csv", index=False, header=False)


import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error

from preprocessing.preprocess_data import preprocess

# === 1. Load Data ===
X_train = preprocess(pd.read_csv("../train_test_splits/train_split.feats.csv"))
X_test = preprocess(pd.read_csv("../train_test_splits/test_split.feats.csv"))
y_train = pd.read_csv("../train_test_splits/train_split.labels.1.csv").squeeze()
y_test = pd.read_csv("../train_test_splits/test_split.labels.1.csv").squeeze()


# === 2. Linear Regression (Baseline) ===
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)
print(f"🔹 Linear Regression MSE: {mse_lin:.2f}")

# === 3. RidgeCV – built-in cross validation to select best alpha ===
#alphas = [0,0.01, 0.1, 1,10, 100, 1000, 10000]

alphas = [0]

ridge_cv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=5)
ridge_cv.fit(X_train, y_train)

# תחזית וביצועים
y_pred_ridge = ridge_cv.predict(X_train)
mse_ridge = mean_squared_error(y_train, y_pred_ridge)

pd.Series(y_pred_ridge).to_csv("ridge_predictions.csv", index=False, header=False)



print(f"✅ Best alpha from RidgeCV: {ridge_cv.alpha_}")
print(f"🔹 RidgeCV MSE on test set: {mse_ridge:.2f}")

