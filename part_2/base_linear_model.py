import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").squeeze()
y_test = pd.read_csv("y_test.csv").squeeze()


lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
y_pred_lin = lin_model.predict(X_test)
mse_lin = mean_squared_error(y_test, y_pred_lin)


ridge_model = Ridge(alpha=1000.0)
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)


print(f"Linear Regression MSE: {mse_lin:.2f}")
print(f"Ridge Regression (alpha=1000) MSE: {mse_ridge:.2f}")

#pd.Series(y_pred_lin).to_csv("linear_predictions.csv", index=False, header=False)
#pd.Series(y_pred_ridge).to_csv("ridge_predictions.csv", index=False, header=False)
