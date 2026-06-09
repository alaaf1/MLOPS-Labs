import json
import os
import pickle
import logging
import dvc.api
import dagshub
import pandas as pd
import mlflow
from typing import Any, Dict
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score
from src.training.mlflow_utils import setup_mlflow, setup_experiment
from src.logger import ExecutorLogger

logger = logging.getLogger(__name__)


def evaluate(client, cfg: Dict[str, Any]):
    test_df = pd.read_parquet(
        os.path.join(cfg["data"]["processed_data_path"], cfg["data"]["test_file"])
    )
    X_test = test_df.drop(cfg["data"]["target_col"], axis=1)
    y_test = test_df[cfg["data"]["target_col"]]

    os.makedirs(cfg["model"]["reports_path"], exist_ok=True)
    all_results = {}

    for name in ["random_forest", "logistic_regression"]:
        model_name = f"titanic-{name}"
        logger.info(f"Evaluating {model_name} from MLflow...")

        version = client.get_latest_versions(name=model_name)[0].version
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")

        preds = model.predict(X_test)

        all_results[name] = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, average="weighted"), 4),
            "recall": round(recall_score(y_test, preds, average="weighted"), 4),
        }
        logger.info(f"{name} -> Accuracy: {all_results[name]['accuracy']}")

    report_path = os.path.join(cfg["model"]["reports_path"], "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"Evaluation report saved to {report_path}")


if __name__ == "__main__":
    logger = ExecutorLogger("dvc-evaluate")
    load_dotenv(".env")
    cfg = dvc.api.params_show()

    dagshub.auth.add_app_token(token=os.getenv("DAGSHUB_TOKEN"))
    dagshub.init(
        repo_owner=os.getenv("DAGSHUB_USERNAME"),
        repo_name=cfg["pipeline"]["model"]["repo_name"],
        mlflow=cfg["pipeline"]["model"]["use_mlflow"]
    )
    setup_experiment(cfg["pipeline"]["model"]["experiment_name"])
    client = setup_mlflow(cfg["pipeline"]["model"]["tracking_uri"], logger)

    evaluate(client, cfg["pipeline"])