import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def process_data(cfg: DictConfig):
    logger.info("Loading raw data...")
    df = pd.read_csv(os.path.join(cfg.raw_data_path, cfg.raw_file))

    df["Title"] = df["Name"].str.extract(r' ([A-Za-z]+)\.')
    rare_titles = df["Title"].value_counts()[
        df["Title"].value_counts() < cfg.rare_title_threshold
    ].index
    df["Title"] = df["Title"].apply(lambda x: "Rare" if x in rare_titles else x)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df.drop(list(cfg.drop_cols), axis=1, inplace=True)

    train_df, test_df = train_test_split(
        df,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=df[cfg.target_col]
    )

    os.makedirs(cfg.processed_data_path, exist_ok=True)
    train_df.to_parquet(os.path.join(cfg.processed_data_path, cfg.train_file))
    test_df.to_parquet(os.path.join(cfg.processed_data_path, cfg.test_file))
    logger.info(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")