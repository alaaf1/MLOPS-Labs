import pandas as pd
import numpy as np
import pickle
import os
import logging

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

logger = logging.getLogger(__name__)

# --- Define features ---
NUMERIC_FEATURES = ["Age", "Fare", "FamilySize", "SibSp", "Parch"]
CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title", "Pclass"]

def build_preprocessor():
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),   # mean for Age as you chose
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),  # handles missing Embarked
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES)
    ])
    return preprocessor

def build_models(preprocessor):
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
    ])

    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])

    return {"random_forest": rf_pipeline, "logistic_regression": lr_pipeline}

def train():
    logger.info("Loading processed data...")
    train_df = pd.read_parquet("data/processed/titanic-train.parquet")

    X_train = train_df.drop("Survived", axis=1)
    y_train = train_df["Survived"]

    preprocessor = build_preprocessor()
    models = build_models(preprocessor)

    os.makedirs("models", exist_ok=True)
    cv_results = {}

    for name, pipeline in models.items():
        logger.info(f"Training {name} with 5-fold CV...")

        scores = cross_validate(
            pipeline, X_train, y_train,
            cv=5,
            scoring="roc_auc",
            n_jobs=-1
        )
        mean_auc = np.mean(scores["test_score"])
        std_auc = np.std(scores["test_score"])
        cv_results[name] = {"mean_auc": mean_auc, "std_auc": std_auc}
        logger.info(f"{name} → CV AUC: {mean_auc:.4f} ± {std_auc:.4f}")

        # Retrain on ALL training data after CV
        pipeline.fit(X_train, y_train)

        # Save the model
        model_path = f"models/{name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(pipeline, f)
        logger.info(f"Model saved to {model_path}")

    return cv_results