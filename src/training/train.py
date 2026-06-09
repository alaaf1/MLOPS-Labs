import pandas as pd
import numpy as np
import pickle
import os
import logging
from omegaconf import DictConfig, OmegaConf
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

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

def build_models(preprocessor, cfg: DictConfig):
    return {
        "random_forest": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", RandomForestClassifier(**OmegaConf.to_container(cfg.random_forest)))
        ]),
        "logistic_regression": Pipeline([
            ("preprocessor", preprocessor),
            ("clf", LogisticRegression(**OmegaConf.to_container(cfg.logistic_regression)))
        ])
    }

import mlflow
import dagshub

def train(cfg=None):
    if cfg is None:
        cfg = OmegaConf.load("conf/config.yaml").pipeline

    # initialize dagshub + mlflow tracking
    dagshub.init(repo_owner="alaaf1", repo_name="MLOPS-Labs-ITI", mlflow=True)
    # or your lab1 repo name
    
    train_df = pd.read_parquet(
        os.path.join(cfg.data.processed_data_path, cfg.data.train_file)
    )
    X_train = train_df.drop(cfg.data.target_col, axis=1)
    y_train = train_df[cfg.data.target_col]

    preprocessor = build_preprocessor()
    models = build_models(preprocessor, cfg.model)

    os.makedirs(cfg.model.models_path, exist_ok=True)

    for name, pipeline in models.items():
        with mlflow.start_run(run_name=name):
            scores = cross_validate(
                pipeline, X_train, y_train,
                cv=cfg.model.cv_folds,
                scoring=cfg.model.scoring,
                n_jobs=-1
            )
            mean_auc = np.mean(scores["test_score"])
            std_auc = np.std(scores["test_score"])

            # log params and metrics
            mlflow.log_params(OmegaConf.to_container(
                cfg.model.random_forest if name == "random_forest" 
                else cfg.model.logistic_regression
            ))
            mlflow.log_metric("cv_mean_auc", mean_auc)
            mlflow.log_metric("cv_std_auc", std_auc)

            pipeline.fit(X_train, y_train)

            # log and register model
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path=name,
                registered_model_name=f"titanic_{name}"  # this registers it
            )

            logger.info(f"{name} -> CV AUC: {mean_auc:.4f} +/- {std_auc:.4f}")