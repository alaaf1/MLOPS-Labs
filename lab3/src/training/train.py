import os
import pickle
import logging
import numpy as np
import pandas as pd
import mlflow
import dagshub
import dvc.api
from typing import Any, Dict
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

from src.training.model_wrapper import ModelWrapper
from src.training.mlflow_utils import setup_mlflow, setup_experiment
from src.logger import ExecutorLogger

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = ["Age", "Fare", "FamilySize", "SibSp", "Parch"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title", "Pclass"]


def build_preprocessor():
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ])


def build_models(preprocessor, cfg: Dict[str, Any]):
    return {
        "random_forest": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", RandomForestClassifier(**cfg["random_forest"]))
        ]),
        "logistic_regression": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(**cfg["logistic_regression"]))
        ])
    }


def train(cfg: Dict[str, Any], client):
    logger.info("Loading processed data...")
    train_df = pd.read_parquet(
        os.path.join(cfg["data"]["processed_data_path"], cfg["data"]["train_file"])
    )
    X_train = train_df.drop(cfg["data"]["target_col"], axis=1)
    y_train = train_df[cfg["data"]["target_col"]]

    preprocessor = build_preprocessor()
    models = build_models(preprocessor, cfg["model"])
    os.makedirs(cfg["model"]["models_path"], exist_ok=True)

    cv_results = {}

    for name, pipeline in models.items():
        logger.info(f"Training {name} with {cfg['model']['cv_folds']}-fold CV...")
        scores = cross_validate(
            pipeline, X_train, y_train,
            cv=cfg["model"]["cv_folds"],
            scoring=cfg["model"]["scoring"],
            n_jobs=-1
        )
        mean_auc = float(np.mean(scores["test_score"]))
        std_auc = float(np.std(scores["test_score"]))
        cv_results[name] = {"mean_auc": round(mean_auc, 4), "std_auc": round(std_auc, 4)}
        logger.info(f"{name} -> CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")

        pipeline.fit(X_train, y_train)

        # Save locally
        model_path = os.path.join(cfg["model"]["models_path"], f"{name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)
        logger.info(f"Model saved to {model_path}")

        # Log to MLflow
        with mlflow.start_run(run_name=name):
            mlflow.log_params({
                **cfg["model"][name.replace("-", "_")],
                "cv_folds": cfg["model"]["cv_folds"]
            })
            mlflow.log_metrics({
                "mean_auc": mean_auc,
                "std_auc": std_auc
            })
            train_preds = pipeline.predict(X_train)
            signature = mlflow.models.infer_signature(X_train, train_preds)
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path=name,
                signature=signature,
                registered_model_name=f"titanic-{name}"
            )
            logger.info(f"{name} logged to MLflow")

    return cv_results


if __name__ == "__main__":
    logger = ExecutorLogger("dvc-train")
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

    train(cfg["pipeline"], client)