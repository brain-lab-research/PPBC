import torch
import numpy as np


def get_loss(loss_cfg, df=None, device=None, loaders=None):
    loss_name = loss_cfg.loss_name
    loss = loss_cfg.config
    if loss_name == "ce":
        pos_weight = calculate_class_weights_multi_class(df)
        pos_weight = torch.tensor(pos_weight).to(device)
        return torch.nn.CrossEntropyLoss(
            weight=pos_weight,
            size_average=loss.size_average,
            ignore_index=loss.ignore_index,
            reduce=loss.reduce,
            reduction=loss.reduction,
            label_smoothing=loss.label_smoothing,
        )
    elif loss_name == "bce":
        if loss.pos_weight is None:
            loss.pos_weight = calculate_pos_weight(df)
        pos_weight = torch.tensor([loss.pos_weight]).to(device)
        return torch.nn.BCEWithLogitsLoss(
            size_average=loss.size_average,
            reduce=loss.reduce,
            reduction=loss.reduction,
            pos_weight=pos_weight,
        )
    else:
        raise ValueError("Unknown type of loss function")


def calculate_pos_weight(df):
    # Convert the 'target' column into a NumPy array
    target_array = np.array(df["target"].tolist())

    # Count zeros and ones for each index
    zeros_count = np.sum(target_array == 0, axis=0)
    ones_count = np.sum(target_array == 1, axis=0)

    # Calculate portion (ratio) of zeros to ones for each index
    ratios = zeros_count / ones_count
    return ratios.tolist()


def calculate_class_weights_multi_class(df):
    df_copy = df.copy()
    target_array = np.array(df_copy["target"].tolist())

    class_weights = {}
    ordered_weights = []

    unique_classes = np.unique(target_array)
    total_count = len(target_array)

    for cls in unique_classes:
        class_count = np.sum(target_array == cls, axis=0)
        class_weight = float(total_count / (len(unique_classes) * class_count))
        class_weights[cls] = class_weight

    # Ensure the weights are added to the list in order of ascending class index
    for cls in sorted(unique_classes):
        ordered_weights.append(class_weights[cls])

    return ordered_weights
