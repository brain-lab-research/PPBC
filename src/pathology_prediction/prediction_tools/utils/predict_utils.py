import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ecglib.data.datasets import EcgDataset
from .data_utils import create_dataframe
from .model_utils import model_inference


def prep_csv_data(cfg: dict, data_key: str, *args, **kwargs) -> tuple:
    """Make pandas.DataFrame contains ECG records info

    :param cfg: cfg file
    :param data_key: Experiment name
    :raises NotImplementedError: _description_

    :return: pandas.DataFrame with ECG records
    """
    print("Read test data...")
    ecg_info = create_dataframe(
        cfg,
        data_key,
        data_filtering=True,
    )

    pathology_names = cfg.task_params.pathology_names
    if cfg.task_params.merge_map:
        pathology_names = cfg.task_params.merge_map.keys()

    num_classes = len(pathology_names)

    return (
        ecg_info,
        list(ecg_info.target),
        pathology_names,
        num_classes,
    )


def predict_on_dataset(
    model: torch.nn.Module,
    dataset: EcgDataset,
    batch_size: int,
    device: str,
    num_workers: int = 0,
    thresholds: list = [0.5],
) -> dict:
    """Makes predictions on input dataset and return results by each threshold from `thresholds`

    :param model: Neural network model
    :param dataset: Evaluation dataset
    :param batch_size: Size of data batch
    :param device: Computing device
    :param num_workers: How many subprocesses to use for data loading, defaults to 0
    :param thresholds: Binarization threshold list, defaults to [0.5]

    :return: A dictionary that contains the prediction results for each binarization threshold
    """
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    results = {}
    raw_preds, probs, true_labels = ([], [], [])

    print("Model inference on input dataset:")
    for batch in tqdm(data_loader, total=len(data_loader)):
        index, (input, targets) = batch

        ecg_signal = input[0]

        # ------------------------------------------- INPUT HANDLER
        inp = ecg_signal.to(device)

        # ------------------------------------------- INPUT HANDLER
        targets = targets.to(device)

        raw, prob, _ = model_inference(model, inp)

        raw_preds.extend(raw.tolist())
        probs.extend(prob.tolist())
        true_labels.extend(targets.tolist())

    results["raw_preds"] = raw_preds
    results["probs"] = probs
    results["true_labels"] = true_labels

    print("Calculation of predictions with threshold list: {}".format(thresholds))
    for thresh in tqdm(thresholds, total=len(thresholds)):
        results[str(thresh)] = (torch.tensor(probs) > thresh).float().tolist()
    return results


def save_model_outputs(
    ecg_info,
    results,
    model_output_path,
):
    output_info = dict(fpath=ecg_info["fpath"], scp_codes=ecg_info["scp_codes"])
    output_info.update(results)
    print("model output is saved to", model_output_path)
    if model_output_path:
        model_output = pd.DataFrame(output_info)
        if os.path.exists(model_output_path):
            current_output = pd.read_csv(model_output_path)
            current_output = current_output.merge(
                model_output, on=["fpath", "scp_codes"]
            )
        else:
            current_output = model_output
        current_output.to_csv(model_output_path, index=False, encoding="utf-8")