import hydra
import pandas as pd
import ast
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from ecglib.data.datasets import EcgDataset
from ecglib.preprocessing.composition import *
import numpy as np
from random import sample


__all__ = [
    "create_dataframe",
    "get_augmentation",
    "get_dataset_loader",
    "make_onehot",
    "merge_columns",
    "filter_by_another_dataframes",
    "get_list_of_clients",
    "change_labels_of_clients",
]


def create_dataframe(
    cfg: dict,
    directories_key: str,
    data_filtering: bool = True,
    dataset: str = "",
) -> tuple:
    """A wrapper for `create_dataframe` with just two parameters

    :param cfg: Configuration
    :param directories_key: Key to extract directories for dataframe (train or test)
    :param data_filtering: Whether to do data filtering

    :return: same as in `create_dataframe` function
    """

    if not dataset:
        dataset = cfg.dataset

    ecg_info = pd.DataFrame()
    for directories in cfg["observed_data_params"][dataset][directories_key]:
        ecg_info = pd.concat([ecg_info, pd.read_csv(directories)])

    # Remove rows with filenames listed in filter_dataframe_files
    ecg_info = filter_by_another_dataframes(
        ecg_info, cfg["observed_data_params"]["filter_dataframe_files"]
    )

    ecg_info = ecg_info[ecg_info[cfg.task_params.task_type].notna()]

    # print(f"Data filtering flag is {data_filtering}")
    if data_filtering:
        # Filter record and patient metadata
        ecg_info["patient_metadata"] = ecg_info.patient_metadata.apply(
            lambda x: ast.literal_eval(x)
        )
        ecg_info["ecg_metadata"] = ecg_info.ecg_metadata.apply(
            lambda x: ast.literal_eval(x)
        )

        # Add age column
        ecg_info["age"] = ecg_info.patient_metadata.apply(lambda x: x["age"])

        ecg_info["patient_metadata"] = ecg_info.patient_metadata.apply(
            lambda x: {
                key: x[key] if isinstance(x[key], list) else [x[key]]
                for key in cfg.ecg_metadata.patient_meta_subset
            }
        )
        ecg_info["ecg_metadata"] = ecg_info.ecg_metadata.apply(
            lambda x: {
                key: x[key] if isinstance(x[key], list) else [x[key]]
                for key in cfg.ecg_metadata.ecg_meta_subset
            }
        )

        # Remove rows with ages that are not inside age_range interval
        ecg_info = ecg_info[
            (
                (ecg_info.age >= cfg.patient_params.age_range[0])
                & (ecg_info.age <= cfg.patient_params.age_range[1])
            )
        ]

        # Remove rows with frequencies that are not listed in observed_frequencies
        ecg_info = ecg_info[
            ecg_info["frequency"].isin(cfg.ecg_record_params.observed_frequencies)
        ]

        # Remove rows with lengths that are not inside ecg_length interval
        ecg_info = ecg_info[
            (
                (ecg_info.ecg_duration >= cfg.ecg_record_params.ecg_length[0])
                & (ecg_info.ecg_duration <= cfg.ecg_record_params.ecg_length[1])
            )
        ]

        if cfg.task_params.validated_by_human:
            if "ptbxl" in cfg.dataset or "ptbxl" in directories_key:
                ecg_info = ecg_info[ecg_info["validated_by_human"]]

    ecg_info.reset_index(drop=True, inplace=True)
    one_hot = make_onehot(
        ecg_info, cfg.task_params.task_type, cfg.task_params.pathology_names
    )

    if cfg.task_params.merge_map:
        one_hot = merge_columns(df=one_hot, merge_map=cfg.task_params.merge_map)

    ecg_info["target"] = one_hot.values.tolist()

    return ecg_info


def get_augmentation(cfg):
    augmentation_transform = cfg.ecg_record_params.augmentation.transforms
    if augmentation_transform:
        aug_list = []
        keys = augmentation_transform.keys()
        for i, key in enumerate(keys):
            augmentation = hydra.utils.instantiate(
                augmentation_transform[key], _convert_="all"
            )
            aug_list.append(augmentation)

        augmentation = Compose(
            transforms=aug_list, p=cfg.ecg_record_params.augmentation.prob
        )
    else:
        augmentation = None
    return augmentation


def get_dataset_loader(
    ecg_info: pd.DataFrame,
    cfg,
    augmentation,
    classes_num,
    drop_last=True,
):
    ecg_target = ecg_info.target.values
    ecg_dataset = EcgDataset.for_train_from_config(
        ecg_info, ecg_target, augmentation, cfg, classes_num
    )
    ecg_loader = DataLoader(
        ecg_dataset,
        batch_size=cfg.training_params.batch_size,
        shuffle=True,
        num_workers=cfg.training_params.num_workers,
        drop_last=drop_last,
    )
    return ecg_loader


def make_onehot(ecg_df, task_type, pathology_names=None):
    """
    Create one_hot vectors for classification

    :param ecg_df: input dataframe
    :param task_type: type of predicted classes (registers, syndromes, etc.)
    :param pathology_names: list of predicted classes

    :return: pandas dataframe
    """
    mlb = MultiLabelBinarizer()
    one_hot = pd.DataFrame(
        mlb.fit_transform(ecg_df[task_type].apply(ast.literal_eval)),
        columns=mlb.classes_,
    )
    if pathology_names:
        drop_cols = set(one_hot.columns) - set(pathology_names)
        one_hot.drop(columns=drop_cols, inplace=True)
    return one_hot


def merge_columns(df, merge_map):
    """
    Logical OR for given one-hot columns

    :param df: input dataframe
    :param merge_map: dictionary: key - name after merge, value - list of columns to be merged

    :return: pandas DataFrame
    """
    for k, v in merge_map.items():
        tmp = df[v].apply(any, axis=1).astype(int)
        df.drop(columns=v, inplace=True)
        df[k] = tmp
    return df


def filter_by_another_dataframes(ecg_info, filter_dataframe_files):
    """
    filter the dataframe by name of files from others

    :param ecg_info: pandas Dataframe
    :param filter_dataframe_files: paths to files containing ecg filenames to be filtered

    :return: filtered Dataframe
    """
    for file_path in filter_dataframe_files:
        try:
            filter_df = pd.read_csv(file_path)
        except pd.errors.EmptyDataError:
            continue
        filter_df_list = filter_df.file_name.to_list()
        ecg_info = ecg_info[~ecg_info.file_name.isin(filter_df_list)]
    return ecg_info


def check_different_clinics(train_value_counts):
    """
    Checks if there are any other clients except from TIS, if yes, these clients are returned
    """

    different_clients = []

    if (
        sum([1 if "ptbxl" in x else 0 for x in train_value_counts.keys()]) != 0
        and sum([1 if "ptbxl" in x else 0 for x in different_clients]) == 0
    ):
        different_clients.extend(["ptbxl_client_0", "ptbxl_client_1"])
    if (
        sum([1 if "sechenovka" in x else 0 for x in train_value_counts.keys()]) != 0
        and sum([1 if "sechenovka" in x else 0 for x in different_clients]) == 0
    ):
        different_clients.extend(["sechenovka_client_0"])

    return different_clients


def get_list_of_clients(
    train_df, min_sample_number, pathology_names, client_amount, client_sample
):
    """
    get list of all clients (hospitals) with at least min_sample_number samples

    :param df: pandas Dataframe
    :param min_sample_number: min number of samples from one client

    :return: list
    """

    train_value_counts = train_df["ID_CLINIC"].value_counts()

    # filter by min_sample_number
    unique_train_values_with_min_occurrences = train_value_counts[
        train_value_counts >= min_sample_number
    ].index.tolist()

    # if there are any clients except from TIS, adds them to unique_train_values_with_min_occurrences, even if they are not that big
    different_clients = check_different_clinics(train_value_counts)
    if different_clients:
        unique_train_values_with_min_occurrences = [
            x
            for x in unique_train_values_with_min_occurrences
            if x not in different_clients
        ][: client_amount - len(different_clients)]
        unique_train_values_with_min_occurrences.extend(different_clients)

    # filter by pathology_names
    unique_train_val_with_min_occ_and_paths = []
    num_clients = client_amount
    for id_clinic in unique_train_values_with_min_occurrences:
        clinic_id_df = train_df[train_df["ID_CLINIC"] == id_clinic]
        df_pathology_mask = clinic_id_df["scp_codes"].apply(
            lambda x: [item in ast.literal_eval(x) for item in pathology_names]
        )
        np_pathology_mask = np.array(df_pathology_mask.values.tolist())
        # check that for all path in pathology_names exists at least one positive sample
        entry_to_df = all(
            [any(np_pathology_mask[:, i]) for i in range(len(pathology_names))]
        )
        if entry_to_df:
            unique_train_val_with_min_occ_and_paths.append(id_clinic)
            num_clients -= 1
        if num_clients == 0:
            break

    # print(unique_train_val_with_min_occ_and_paths)
    # assert num_clients == 0, f"min_sample_number or pathology_names filters made the number of clients < {num_clients}"
    unique_train_val_with_min_occ_and_paths = np.array(
        unique_train_val_with_min_occ_and_paths
    )[
        sample(
            range(0, len(unique_train_val_with_min_occ_and_paths)),
            int(client_sample * len(unique_train_val_with_min_occ_and_paths)),
        )
    ]
    return unique_train_val_with_min_occ_and_paths


def change_labels_of_clients(
    df, method, list_of_clients=None, attacking_clients=None, p=0.5, dataset_type="ecg"
):
    if dataset_type == "ecg":
        client_type = "ID_CLINIC"
    elif dataset_type == "cifar":
        client_type = "client"

    if method == "label_flipping":
        for num in attacking_clients:
            client_targets = df[df[client_type] == list_of_clients[num]]["target"]
            labels = np.array(client_targets.tolist())
            attacked_labels = np.random.choice(np.prod(labels.shape), int(p * np.prod(labels.shape)), replace=False)
            if dataset_type == "ecg":
                labels.flat[attacked_labels] -= 1
            elif dataset_type == "cifar":
                corrupted_labels = np.random.randint(0, 10, size=attacked_labels.size)
                labels.flat[attacked_labels] = corrupted_labels
            target_column = pd.Series(
                np.abs(labels).tolist(), name="target", index=client_targets.index
            )
            df.update(target_column)
    elif method == "random_rounds_random_clients":
        labels = np.array(df["target"].tolist())
        attacked_labels = np.random.choice(np.prod(labels.shape), int(p * np.prod(labels.shape)), replace=False)
        if dataset_type == "ecg":
            labels.flat[attacked_labels] -= 1
        elif dataset_type == "cifar":
            corrupted_labels = np.random.randint(0, 10, size=attacked_labels.size)
            labels.flat[attacked_labels] = corrupted_labels
        target_column = pd.Series(
            np.abs(labels).tolist(), name="target", index=df["target"].index
        )
        df.update(target_column)
    # change types due to df.update 
    if dataset_type == "cifar":
        df['target'] = df['target'].astype(int)
    return df
