import math
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    fbeta_score,
)

from .cifar_utils import calculate_cifar_metrics


__all__ = [
    "compute_metrics",
    "metrics_report",
    "print_metrics",
    "select_best_validation_threshold",
    "stopping_criterion",
]


def compute_metrics(
    target,
    prediction,
    pathology_names,
    probs,
):
    """
    Compute metrics from input predictions and
    ground truth labels

    :param target: Ground truth labels
    :param prediction: Predicted labels
    :param pathology_names: Class names of pathologies
    :return: pandas DataFrame with metrics and confusion matrices for each class
    """
    df = pd.DataFrame(
        columns=pathology_names,
        index=[
            "Specificity",
            "Sensitivity",
            "G-mean",
            "f1-score",
            "fbeta2-score",
            "ROC-AUC",
            "AP",
            "Precision (PPV)",
            "NPV",
        ],
    )
    conf_mat_df = pd.DataFrame(columns=["TN", "FP", "FN", "TP"], index=pathology_names)
    target = np.array(target, int)
    prediction = np.array(prediction, int)
    probs = np.array(probs)

    for i, col in enumerate(pathology_names):
        try:
            tn, fp, fn, tp = confusion_matrix(target[:, i], prediction[:, i]).ravel()
            df.loc["Specificity", col] = tn / (tn + fp)
            df.loc["Sensitivity", col] = tp / (tp + fn)
            df.loc["G-mean", col] = math.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))
            df.loc["f1-score", col] = f1_score(target[:, i], prediction[:, i])
            df.loc["fbeta2-score", col] = fbeta_score(
                target[:, i], prediction[:, i], beta=2
            )
            df.loc["ROC-AUC", col] = roc_auc_score(target[:, i], probs[:, i])
            df.loc["AP", col] = average_precision_score(target[:, i], prediction[:, i])
            df.loc["Precision (PPV)", col] = tp / (tp + fp)
            df.loc["NPV", col] = 0 if tn + fn == 0 else tn / (tn + fn)

            conf_mat_df.loc[col] = [tn, fp, fn, tp]
        except ValueError:
            raise
    return df, conf_mat_df


def metrics_report(
    targets: np.ndarray,
    bin_preds: np.ndarray,
    pathology_names: list,
    probs: np.ndarray,
    verbose: bool = True,
) -> tuple:
    """Returns metrics (ROC-AUC, AP, f1-score, Specificity, Sensitivity, Precision) and confusion matrix

    :param targets: True labels
    :type targets: numpy.ndarray
    :param bin_preds: Predicted labels
    :type bin_preds: numpy.ndarray
    :param pathology_names: Pathology class names
    :type pathology_names: list
    :param verbose: Display metrics in console, defaults to True
    :type verbose: bool, optional
    :return: Metrics and confusion matrix
    :rtype: tuple
    """
    metrics, conf_matrix = compute_metrics(targets, bin_preds, pathology_names, probs)
    if verbose:
        print(metrics)
        print(conf_matrix)
        print(classification_report(targets, bin_preds, zero_division=False))
    return metrics, conf_matrix


def print_metrics(results, thresholds, pathology_names):
    print("=============METRICS REPORT=============")
    metrics_dict = {}
    for t in thresholds:
        print("Metrics with `threshold = {}`".format(t))
        metrics, conf_matrix = metrics_report(
            results["true_labels"],
            results[str(t)],
            pathology_names,
            results["probs"],
            verbose=True,
        )
        sub_dict = {"metrics": metrics, "confusion_matrix": conf_matrix}
        metrics_dict[str(t)] = sub_dict
    return metrics_dict


def select_best_validation_threshold(
    fin_targets,
    fin_outputs,
    metrics_threshold,
    prediction_threshold,
):
    if prediction_threshold != None:
        return prediction_threshold
    assert metrics_threshold in ["gmean", "f1-score", "ROC-AUC"]
    M = len(fin_targets[0])  # Number of classes
    thresholds = np.arange(-0.01, 0.9, 0.06)
    best_thresholds = []

    for class_idx in range(M):
        tpr = []
        fpr = []
        precision = []
        positive_samples = sum(
            [1 for targets in fin_targets if targets[class_idx] == 1]
        )
        for w in thresholds:
            outputs = torch.where(
                fin_outputs[:, class_idx] > w, torch.tensor(1.0), torch.tensor(0.0)
            )
            positive_outputs = list(outputs).count(1)
            if positive_outputs == 0:
                break
            tp = sum(
                1
                for yp, yt in zip(outputs, fin_targets)
                if yp == 1.0 and yt[class_idx] == 1.0
            )
            fp = sum(
                1
                for yp, yt in zip(outputs, fin_targets)
                if yp == 1.0 and yt[class_idx] != 1.0
            )
            tpr.append(tp / positive_samples)
            fpr.append(fp / (len(fin_targets) - positive_samples))
            precision.append(
                sum(
                    1
                    for yp, yt in zip(outputs, fin_targets)
                    if yp == 1.0 and yt[0] == 1.0
                )
                / positive_outputs
            )

        if metrics_threshold == "gmean":
            metric = np.sqrt(np.array(tpr) * (1 - np.array(fpr)))
        if metrics_threshold == "f1-score":
            metric = 2 * (
                np.array(tpr)
                * np.array(precision)
                / (np.array(tpr) + np.array(precision))
            )
        if metrics_threshold == "ROC-AUC":
            metric = (np.array(tpr) + (1 - np.array(fpr))) / 2

        best_threshold_idx = np.argmax(metric)
        best_thresholds.append(thresholds[best_threshold_idx])

    # print("best thresholds:", best_thresholds)
    return torch.tensor(best_thresholds)


def stopping_criterion(
    val_loss,
    metrics,
    best_metrics,
    epochs_no_improve,
):
    """
    Define stopping criterion for metrics from config['saving_metrics']
    best_metrics is updated only if every metric from best_metrics.keys() has improved

    :param val_loss: validation loss
    :param metrics: validation metrics
    :param best_metrics: the best metrics for the current epoch
    :param epochs_no_improve: number of epochs without best_metrics updating

    :return: epochs_no_improve, best_metrics
    """
    # get average metrics by class
    metrics = dict(metrics.mean(axis=1))
    # define condition best_metric >= metric for all except for loss
    metrics_mask = all(
        metrics[key] >= best_metrics[key] for key in best_metrics.keys() - {"loss"}
    )
    if not metrics_mask:
        epochs_no_improve += 1
        return epochs_no_improve, best_metrics
    if "loss" in list(best_metrics.keys()):
        if val_loss >= best_metrics["loss"]:
            epochs_no_improve += 1
            return epochs_no_improve, best_metrics
    # Updating best_metrics
    for key in list(best_metrics.keys()):
        if key == "loss":
            best_metrics[key] = val_loss
        else:
            best_metrics[key] = metrics[key]
    # Updating epochs_no_improve
    epochs_no_improve = 0
    return epochs_no_improve, best_metrics


def calculate_metrics(
    fin_targets,
    fin_outputs,
    metrics_threshold,
    pathology_names,
    path_example,
    prediction_threshold,
    verbose=False,
):
    if "image" in path_example:
        # Get results
        softmax = torch.nn.Softmax(dim=1)
        results = softmax(torch.as_tensor(fin_outputs)).max(dim=1)[1]
        fin_targets = torch.as_tensor(fin_targets)
        # Calc metrics
        metrics = calculate_cifar_metrics(fin_targets, results, verbose)
        prediction_threshold = None
    else:
        # Get results
        sigmoid = torch.nn.Sigmoid()
        fin_outputs = sigmoid(torch.as_tensor(fin_outputs))
        prediction_threshold = select_best_validation_threshold(
            fin_targets, fin_outputs, metrics_threshold, prediction_threshold
        )
        results = (fin_outputs > prediction_threshold).float()
        # Calc metrics
        metrics, _ = metrics_report(
            fin_targets, results.tolist(), pathology_names, fin_outputs, verbose
        )
    return metrics, prediction_threshold


def check_metrics_names(metrics):
    allowed_metrics = [
        "loss",
        "Specificity",
        "Sensitivity",
        "G-mean",
        "f1-score",
        "fbeta2-score",
        "ROC-AUC",
        "AP",
        "Precision (PPV)",
        "NPV",
    ]

    assert all(
        [k in allowed_metrics for k in metrics.keys()]
    ), f"federated_params.server_saving_metrics can be only {allowed_metrics}, but get {metrics.keys()}"

    if list(metrics.keys()) != ["loss"]:
        not_loss_metrics = [k for k in metrics.keys() if k != "loss"]
        print(
            f"""Server best model updating based on {not_loss_metrics} metrics in ECG setup requires sending appropriate threshold to clients.\n
                At the current moment, this functionality are not supported.\n
                If you want to manage threshold, you can set up it in `training_params.prediction_threshold` config."""
        )
