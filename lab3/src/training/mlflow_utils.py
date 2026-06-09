import mlflow


def setup_mlflow(tracking_uri: str, logger=None):
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.client.MlflowClient(tracking_uri=tracking_uri)
    if logger:
        logger.info("MLflow client set up successfully.")
    return client


def setup_experiment(experiment_name: str):
    if not mlflow.get_experiment_by_name(experiment_name):
        mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)