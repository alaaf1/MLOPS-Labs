import duckdb
import mlflow
import dagshub
import pandas as pd
from datetime import datetime
from prefect import flow, task
import os
from dotenv import load_dotenv

load_dotenv()
MOTHERDUCK_TOKEN = os.environ.get("motherduck_token" )


@task(name="extract-data", log_prints=True)
def extract_data() -> pd.DataFrame:
    print("Extracting data from MotherDuck...")
    # open connection, query, and close immediately — don't keep it open
    with duckdb.connect(f"md:my_db?motherduck_token={MOTHERDUCK_TOKEN}") as conn:
        df = conn.execute("SELECT * FROM test").df()
    print(f"Extracted {len(df)} rows")
    return df

@task(name="transform-data", log_prints=True)
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Transforming data...")
    
    # recreate the same features from process_data.py
    df["Title"] = df["Name"].str.extract(r' ([A-Za-z]+)\.')
    rare_titles = df["Title"].value_counts()[df["Title"].value_counts() < 10].index
    df["Title"] = df["Title"].apply(lambda x: "Rare" if x in rare_titles else x)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    
    # drop columns the model doesn't expect
    df.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1, inplace=True, errors="ignore")
    
    print(f"Columns after transform: {list(df.columns)}")
    return df

@task(name="load-model", log_prints=True)
def load_model():
    print("Loading model from DagsHub MLflow Registry...")
    dagshub.init(repo_owner="alaaf1", repo_name="MLOPS-Labs-ITI", mlflow=True)
    
    # load the Production model from registry
    model = mlflow.sklearn.load_model(
        model_uri="models:/titanic_random_forest/1"
    )
    print("Model loaded successfully")
    return model

@task(name="predict", log_prints=True)
def predict(model, df: pd.DataFrame) -> pd.DataFrame:
    print("Running predictions...")
    predictions = model.predict(df)
    probabilities = model.predict_proba(df)[:, 1]
    
    df = df.copy()
    df["predicted_survived"] = predictions
    df["survival_probability"] = probabilities.round(4)
    df["predicted_at"] = datetime.now().isoformat()
    
    print(f"Generated {len(predictions)} predictions")
    return df

@task(name="save-predictions", log_prints=True)
def save_predictions(df: pd.DataFrame):
    print("Saving predictions to MotherDuck...")
    with duckdb.connect(f"md:my_db?motherduck_token={MOTHERDUCK_TOKEN}") as conn:
        conn.execute("DROP TABLE IF EXISTS titanic_predictions")
        conn.execute("CREATE TABLE titanic_predictions AS SELECT * FROM df")
    print(f"Saved {len(df)} predictions to titanic_predictions table")

@flow(name="titanic-batch-prediction", log_prints=True)
def batch_prediction_flow():
    # orchestrate the tasks in order
    raw_df = extract_data()
    clean_df = transform_data(raw_df)
    model = load_model()
    predictions_df = predict(model, clean_df)
    save_predictions(predictions_df)
    print("Batch job completed successfully")

if __name__ == "__main__":
    batch_prediction_flow()