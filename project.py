from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import json
import mlflow

OWNER_NAME = "Sofia Lisichkina"

def init_metrics(**kwargs):
    experiment_name = "Sofia Lisichkina"
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="@s_sophi") as parent_run:
        run_id = parent_run.info.run_id
        experiment_id = parent_run.info.experiment_id
    
    model_name = kwargs['model_name']
    timestamp = datetime.now().isoformat()
    metrics = {
        'init': {
            'timestamp': timestamp,
            'model_name': model_name,
            'experiment_id': experiment_id,
            'run_id': run_id
        }
    }
    return metrics

def get_data(**kwargs):
    ti = kwargs['ti']
    start_time = time.time()
    data = fetch_california_housing(as_frame=True)
    end_time = time.time()

    df = data.frame
    dataset_size = df.shape

    bucket = Variable.get("S3_BUCKET")
    s3_hook = S3Hook("s3_connection")
    path = f"{OWNER_NAME}/datasets/data.csv"
    s3_hook.load_string(df.to_csv(index=False), key=path, bucket_name=bucket, replace=True)

    metrics = ti.xcom_pull(task_ids='init')
    metrics['get_data'] = {
        'start_time': start_time,
        'end_time': end_time,
        'dataset_size': dataset_size
    }
    return metrics

def prepare_data(**kwargs):
    ti = kwargs['ti']
    start_time = time.time()

    bucket = Variable.get("S3_BUCKET")
    s3_hook = S3Hook("s3_connection")
    path = f"{OWNER_NAME}/datasets/data.csv"
    data = s3_hook.read_key(key=path, bucket_name=bucket)
    df = pd.read_csv(pd.compat.StringIO(data))

    X = df.drop('MedHouseVal', axis=1)
    y = df['MedHouseVal']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    features = X.columns.tolist()

    prepared_df = pd.DataFrame(X_scaled, columns=features)
    prepared_df['MedHouseVal'] = y
    path_prepared = f"{OWNER_NAME}/datasets/prepared_data.csv"
    s3_hook.load_string(prepared_df.to_csv(index=False), key=path_prepared, bucket_name=bucket, replace=True)

    end_time = time.time()

    metrics = ti.xcom_pull(task_ids='get_data')
    metrics['prepare_data'] = {
        'start_time': start_time,
        'end_time': end_time,
        'features': features
    }
    return metrics

def train_model(model_name, **kwargs):
    ti = kwargs['ti']
    models = {
        "LinearRegression": LinearRegression,
        "DecisionTreeRegressor": DecisionTreeRegressor,
        "RandomForestRegressor": RandomForestRegressor
    }

    model_class = models.get(model_name)
    experiment_id, parent_run_id = ti.xcom_pull(task_ids='init')['init']['experiment_id'], ti.xcom_pull(task_ids='init')['init']['run_id']

    with mlflow.start_run(experiment_id=experiment_id, run_id=parent_run_id, nested=True):
        start_time = time.time()

        bucket = Variable.get("S3_BUCKET")
        s3_hook = S3Hook("s3_connection")
        path_prepared = f"{OWNER_NAME}/datasets/prepared_data.csv"
        data = s3_hook.read_key(key=path_prepared, bucket_name=bucket)
        prepared_df = pd.read_csv(pd.compat.StringIO(data))

        X = prepared_df.drop('MedHouseVal', axis=1)
        y = prepared_df['MedHouseVal']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = model_class()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, model_name)

        end_time = time.time()

        metrics = ti.xcom_pull(task_ids='prepare_data')
        metrics['train_model'] = {
            'start_time': start_time,
            'end_time': end_time,
            'mse': mse,
            'r2': r2
        }
        return metrics

def save_results(**kwargs):
    ti = kwargs['ti']
    metrics = ti.xcom_pull(task_ids=kwargs['prev_task'])
    
    bucket = Variable.get("S3_BUCKET")
    s3_hook = S3Hook("s3_connection")
    path_results = f"{OWNER_NAME}/results/metrics.json"
    s3_hook.load_string(json.dumps(metrics, indent=4), key=path_results, bucket_name=bucket, replace=True)

    return "Metrics saved successfully"

default_args = {
    'owner': OWNER_NAME,
    'retries': 3,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'SofiaLisichkina',
    default_args=default_args,
    description='A DAG to train three models with MLFlow logging',
    schedule_interval='0 1 * * *',  # Daily at 01:00
    start_date=datetime(2023, 1, 1),
    tags=['mlops'],
)

with dag:
    init_task = PythonOperator(
        task_id='init',
        python_callable=init_metrics,
        op_kwargs={'model_name': 'BaseModel'},  # Placeholder name
        dag=dag,
    )

    get_data_task = PythonOperator(
        task_id='get_data',
        python_callable=get_data,
        provide_context=True,
        dag=dag,
    )

    prepare_data_task = PythonOperator(
        task_id='prepare_data',
        python_callable=prepare_data,
        provide_context=True,
        dag=dag,
    )

    model_names = ["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor"]

    for model_name in model_names:
        train_model_task = PythonOperator(
            task_id=f'train_model_{model_name}',
            python_callable=train_model,
            provide_context=True,
            op_kwargs={'model_name': model_name},
            dag=dag,
        )

        save_results_task = PythonOperator(
            task_id=f'save_results_{model_name}',
            python_callable=save_results,
            provide_context=True,
            op_kwargs={'prev_task': f'train_model_{model_name}'},
            dag=dag,
        )

        # Connect tasks
        init_task >> get_data_task >> prepare_data_task >> train_model_task >> save_results_task