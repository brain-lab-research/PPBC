import torch
import hydra
from ecglib.models.model_builder import create_model, Combination

# from torchvision.models import resnet18
from .resnet import ResNet18


__all__ = [
    "get_model",
    "freeze_layers_cnn",
    "model_inference",
]


def get_model(
    cfg,
    leads_num,
):

    if cfg.dataset == "cifar":
        model = ResNet18(num_classes=10)
        if cfg.training_params.device == "cuda":
            device = "{}:{}".format(
                cfg.training_params.device, cfg.training_params.device_ids[0]
            )
            model.to(device)
        return model

    combination = Combination.from_string(cfg.models_combination)
    assert combination in [Combination.SINGLE, Combination.CNNTAB]
    models = [model["model_name"] for model in cfg.models]
    configs = [hydra.utils.instantiate(model["config"]) for model in cfg.models]
    pretrain_path = (
        cfg.predict.checkpoint_path if cfg.training_params.pretrained else None
    )

    pathologies = cfg.task_params.pathology_names
    layers_to_freeze = cfg.training_params.num_freezes_layers

    model = create_model(
        model_name=models,
        config=configs,
        combine=combination,
        pretrained=cfg.training_params.pretrained,
        pretrained_path=pretrain_path,
        pathology=pathologies,
        leads_count=leads_num,
        num_classes=cfg.task_params.classes,
    )

    if layers_to_freeze > 0:
        freeze_layers_cnn(model, layers_to_freeze)

    if cfg.training_params.device == "cuda":
        device = "{}:{}".format(
            cfg.training_params.device, cfg.training_params.device_ids[0]
        )
        model.to(device)

    return model


def freeze_layers_cnn(model: torch.nn.Module, count_to_freeze: int) -> None:
    counter = count_to_freeze
    for block in model.children():
        for child in block.children():
            if counter > 0:
                for param in child.parameters():
                    param.requires_grad = False
                counter -= 1
    return model


def model_inference(
    model: torch.nn.Module,
    inp_data: torch.Tensor,
    threshold: float = 0.5,
) -> tuple:
    """Makes predictions using given model on input data

    :param model: Neural network model
    :param inp_data: Input data tensor
    :param threshold: Binarization threshold

    :return: Tuple of raw and binarized predictions
    """
    with torch.no_grad():
        outputs = model(inp_data)
    return (
        outputs,
        torch.nn.Sigmoid()(outputs),
        (torch.nn.Sigmoid()(outputs) > threshold).float(),
    )
