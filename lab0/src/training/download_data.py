import kagglehub
import shutil
import os
import logging

logger = logging.getLogger(__name__)
os.environ["KAGGLE_USERNAME"] = "alaafaisal"
os.environ["KAGGLE_KEY"] = "e28fa081748fc22e8bbabfa358bc8577"
def download_data() :
    
    logger.info("Downloading Titanic dataset from Kaggle...")
    source_path = kagglehub.competition_download("titanic")
    destination_path = "data/raw"
    os.makedirs(destination_path, exist_ok=True)

    for file in os.listdir(source_path):
        shutil.copyfile(
            os.path.join(source_path, file),
            os.path.join(destination_path, file)
        )
    logger.info(f"Data saved to {destination_path}")