import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os

logger = logging.getLogger(__name__)

def process_data():
    logger.info("Loading raw data...")
    df = pd.read_csv("data/raw/train.csv")

    # --- Feature Engineering ---
    # Extract title from name
    df["Title"] = df["Name"].str.extract(r' ([A-Za-z]+)\.')
    # Simplify rare titles
    rare_titles = df["Title"].value_counts()[df["Title"].value_counts() < 10].index
    df["Title"] = df["Title"].apply(lambda x: "Rare" if x in rare_titles else x)

    # Family size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # Drop columns we won't use
    df.drop(["Name", "Ticket", "Cabin", "PassengerId"], axis=1, inplace=True)

    # --- Split ---
    train_df, test_df = train_test_split(
        df,
        test_size=0.15,
        random_state=42,
        stratify=df["Survived"]
    )

    # --- Save ---
    processed_path = "data/processed"
    os.makedirs(processed_path, exist_ok=True)
    train_df.to_parquet(os.path.join(processed_path, "titanic-train.parquet"))
    test_df.to_parquet(os.path.join(processed_path, "titanic-test.parquet"))
    logger.info(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")