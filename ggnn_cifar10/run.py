import dotenv
import hydra
from omegaconf import DictConfig
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
dotenv.load_dotenv(override=True)


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    import time
    import traceback

    import wandb

    from src.train import train
    from src.utils import utils

    utils.extras(config)

    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    # Train model
    try:
        return train(config)
    except Exception:
        traceback.print_exc()
        wandb.finish()
        time.sleep(180)
        try:
            return train(config)
        except Exception:
            wandb.finish()
            time.sleep(120)
            return 0


if __name__ == "__main__":
    main()
