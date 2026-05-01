import logging
import sys
import dvc.api

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
    logger = logging.getLogger(__name__)
    logger.info("Training started")

    cfg = dvc.api.params_show()

    logger.info(f"Pipeline Parameters:\n{cfg['pipeline']}")

    download_data(cfg["pipeline"]["data"])
    process_data(cfg["pipeline"]["data"])
    cv_results = train(cfg["pipeline"])
    evaluate(cfg["pipeline"])

    logger.info("Training finished")