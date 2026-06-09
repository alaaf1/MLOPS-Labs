import kagglehub
import shutil
import os
import logging
from typing import Any, Dict
import dvc.api

logger = logging.getLogger(__name__)

os.environ["KAGGLE_USERNAME"] = "alaafaisal"
os.environ["KAGGLE_KEY"] = "e28fa081748fc22e8bbabfa358bc8577"


def download_data(cfg: Dict[str, Any]):
    logger.info("Downloading Titanic dataset from Kaggle...")
    source_path = kagglehub.competition_download(cfg["kaggle_competition"])
    os.makedirs(cfg["raw_data_path"], exist_ok=True)
    for file in os.listdir(source_path):
        shutil.copyfile(
            os.path.join(source_path, file),
            os.path.join(cfg["raw_data_path"], file)
        )
    logger.info(f"Data saved to {cfg['raw_data_path']}")
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    cfg = dvc.api.params_show()
    download_data(cfg["pipeline"]["data"])