import mlflow
import mlflow.sklearn
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Определение эксперимента
experiment_name = "Sofia Lisichkina"
experiment_id = mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# Загрузка данных
def load_and_prepare_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Функция для обучения модели
def train_and_log_model(model_class, model_name):
    with mlflow.start_run(run_name=model_name, nested=True) as run:
        X_train, X_test, y_train, y_test = load_and_prepare_data()
        
        model = model_class()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"{model_name} - MSE: {mse}, R2: {r2}")
        
        # Логирование метрик
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        # Логирование модели
        mlflow.sklearn.log_model(model, model_name)

# Начальный parent run
with mlflow.start_run(run_name="@s_sophi", experiment_id=experiment_id) as parent_run:
    train_and_log_model(LinearRegression, "Linear Regression")
    train_and_log_model(DecisionTreeRegressor, "Decision Tree Regressor")
    train_and_log_model(RandomForestRegressor, "Random Forest Regressor")

print(f"Experiment URL: {mlflow.get_artifact_uri()}")