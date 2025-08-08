import mlflow
import time

def log_login(username):
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Patent_RAG_User_Logins")
        with mlflow.start_run(run_name=username):
            mlflow.log_param("username", username)
            mlflow.log_param("login_time", time.strftime("%Y-%m-%d %H:%M:%S"))
    except Exception as e:
        print(f"⚠️ MLflow logging failed: {e}")
