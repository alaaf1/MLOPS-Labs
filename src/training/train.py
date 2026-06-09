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

def train(cfg: DictConfig):
    logger.info("Loading processed data...")
    train_df = pd.read_parquet(
        os.path.join(cfg.data.processed_data_path, cfg.data.train_file)
    )
    X_train = train_df.drop(cfg.data.target_col, axis=1)
    y_train = train_df[cfg.data.target_col]

    preprocessor = build_preprocessor()
    models = build_models(preprocessor, cfg.model)

    os.makedirs(cfg.model.models_path, exist_ok=True)
    cv_results = {}

    for name, pipeline in models.items():
        logger.info(f"Training {name} with {cfg.model.cv_folds}-fold CV...")
        scores = cross_validate(
            pipeline, X_train, y_train,
            cv=cfg.model.cv_folds,
            scoring=cfg.model.scoring,
            n_jobs=-1
        )
        mean_auc = np.mean(scores["test_score"])
        std_auc = np.std(scores["test_score"])
        cv_results[name] = {"mean_auc": round(mean_auc, 4), "std_auc": round(std_auc, 4)}
        logger.info(f"{name} -> CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")

        pipeline.fit(X_train, y_train)
        model_path = os.path.join(cfg.model.models_path, f"{name}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)
        logger.info(f"Model saved to {model_path}")

    return cv_results