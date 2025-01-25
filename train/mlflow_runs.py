import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_wine_quality_data(file_path='winequality.csv'):
    data = pd.read_csv(file_path)
    data = data.drop("wine_type", axis=1)
    X = data.drop('quality', axis=1)
    y = data['quality']
    return X, y


mlflow.set_experiment('wine_quality_prediction')


def train_and_log_model(X_train, X_test, y_train, y_test, params=None):
    with mlflow.start_run():
        if params:
            for param, value in params.items():
                mlflow.log_param(param, value)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LinearRegression(**params) if params else LinearRegression()
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metrics({
            'mse': mse,
            'mae': mae,
            'r2': r2
        })

        mlflow.sklearn.log_model(model, 'linear_regression_model')

        return model, mse, mae, r2


X, y = load_wine_quality_data()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

param_configs = [
    None,
    {'fit_intercept': False},
    {'copy_X': False}
]

for params in param_configs:
    train_and_log_model(X_train, X_test, y_train, y_test, params)
