import logging
import sys
import hydra
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

from src.training.process_data import process_data
from src.training.train import train
from src.training.evaluate import evaluate
from src.training.download_data import download_data

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger = logging.getLogger(__name__)
    logger.info("Training started")
    logger.info(f"Pipeline Parameters:\n{OmegaConf.to_yaml(cfg)}")

    download_data(cfg.pipeline.data)
    process_data(cfg.pipeline.data)
    cv_results = train(cfg.pipeline)
    evaluate(cfg.pipeline)

    logger.info("Training finished")

if __name__ == "__main__":
    main()