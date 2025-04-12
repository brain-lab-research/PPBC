import hydra

from omegaconf import DictConfig

from utils.data_utils import create_dataframe
from engine import select_train


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train_model(cfg: DictConfig):
    print("Checkpoint path: ", cfg.single_run_dir)
    print(f"Read train_directories data...")
    train_df = create_dataframe(cfg, "train_directories")
    print(f"Read valid_directories data...")
    valid_df = create_dataframe(cfg, "valid_directories")
    select_train(cfg, train_df, valid_df)


if __name__ == "__main__":
    train_model()
