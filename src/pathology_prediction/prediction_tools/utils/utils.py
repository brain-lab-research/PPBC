import torch


__all__ = [
    "extract_from_checkpoint",
    "get_correct_device",
]


def extract_from_checkpoint(checkpoint_path: str) -> tuple:
    """Extract stored information from checkpoint

    :param checkpoint_path: Path to checkpoint file
    :type checkpoint_path: str

    :return: Extracted git, state_dict, config and metrics
    """

    model_info = torch.load(checkpoint_path, map_location=torch.device("cpu"))

    try:
        # Backward compatibility with old checkpoints with no git information
        git = model_info["git"]
    except KeyError:
        print(f"No git information in the checkpoint {checkpoint_path}")
        git = None

    state_dict = model_info["model"]
    config = model_info["config_file"]
    
    try:
        metrics = model_info["metrics"]
    except KeyError:
        print(f"No validation metrics information in the checkpoint {checkpoint_path}")
        metrics = None

    return (git, state_dict, config, metrics)


def get_correct_device(cfg: dict) -> str:
    if cfg.training_params.device == "cuda":
        return "{}:{}".format(cfg.training_params.device, cfg.training_params.device_ids[0]) 
    else:
        return cfg.training_params.device

