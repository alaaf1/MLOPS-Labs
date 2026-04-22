import logging
import sys
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

from src.training.download_data import download_data
from src.training.process_data import process_data
from src.training.train import train
from src.training.evaluate import evaluate

if __name__ == "__main__":
    download_data()
    process_data()
    cv_results = train()
    evaluate()