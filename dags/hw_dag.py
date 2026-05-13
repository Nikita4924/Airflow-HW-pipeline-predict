import datetime as dt
import os

from airflow.models import DAG
from airflow.operators.python import PythonOperator

path = "/opt/airflow/dags"
os.environ['PROJECT_PATH'] = path

from modules.pipeline import pipeline
from modules.predict import predict

args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 6, 10),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}

with DAG(
        dag_id='car_price_prediction',
        schedule="00 15 * * *",
        default_args=args,
) as dag:
    pipeline_task = PythonOperator(
        task_id='pipeline',
        python_callable=pipeline,
    )

    predict_task = PythonOperator(
        task_id='predict',
        python_callable=predict,
    )

    pipeline_task >> predict_task