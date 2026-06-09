import kagglehub
import shutil
import os
import logging
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
os.environ["KAGGLE_USERNAME"] = "alaafaisal"
os.environ["KAGGLE_KEY"] = "e28fa081748fc22e8bbabfa358bc8577"
def download_data(cfg: DictConfig):
    logger.info("Downloading Titanic dataset from Kaggle...")
    source_path = kagglehub.competition_download(cfg.kaggle_competition)
    os.makedirs(cfg.raw_data_path, exist_ok=True)
    for file in os.listdir(source_path):
        shutil.copyfile(
            os.path.join(source_path, file),
            os.path.join(cfg.raw_data_path, file)
        )
    logger.info(f"Data saved to {cfg.raw_data_path}")