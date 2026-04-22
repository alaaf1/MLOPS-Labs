import pandas as pd
import pickle
import json
import os
import logging

from sklearn.metrics import classification_report, roc_auc_score

logger = logging.getLogger(__name__)

MODEL_NAMES = ["random_forest", "logistic_regression"]

def evaluate():
    test_df = pd.read_parquet("data/processed/titanic-test.parquet")
    X_test = test_df.drop("Survived", axis=1)
    y_test = test_df["Survived"]

    os.makedirs("reports", exist_ok=True)
    all_results = {}

    for name in MODEL_NAMES:
        logger.info(f"Evaluating {name}...")

        with open(f"models/{name}.pkl", "rb") as f:
            model = pickle.load(f)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        report = classification_report(y_test, y_pred, output_dict=True)
        auc = roc_auc_score(y_test, y_prob)

        all_results[name] = {
            "roc_auc": round(auc, 4),
            "classification_report": report
        }
        logger.info(f"{name} → Test AUC: {auc:.4f}")

    # Save report
    report_path = "reports/evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"Evaluation report saved to {report_path}")