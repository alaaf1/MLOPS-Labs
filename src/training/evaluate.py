import pandas as pd
import pickle
import json
import os
import logging
from omegaconf import DictConfig
from sklearn.metrics import classification_report, roc_auc_score

logger = logging.getLogger(__name__)

MODEL_NAMES = ["random_forest", "logistic_regression"]

def evaluate(cfg: DictConfig):
    test_df = pd.read_parquet(
        os.path.join(cfg.data.processed_data_path, cfg.data.test_file)
    )
    X_test = test_df.drop(cfg.data.target_col, axis=1)
    y_test = test_df[cfg.data.target_col]

    os.makedirs(cfg.model.reports_path, exist_ok=True)
    all_results = {}

    for name in MODEL_NAMES:
        logger.info(f"Evaluating {name}...")
        with open(os.path.join(cfg.model.models_path, f"{name}.pkl"), "rb") as f:
            model = pickle.load(f)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)

        all_results[name] = {"roc_auc": round(auc, 4), "classification_report": report}
        logger.info(f"{name} -> Test AUC: {auc:.4f}")

    report_path = os.path.join(cfg.model.reports_path, "evaluation_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"Evaluation report saved to {report_path}")