import os
import ast
import hydra
import numpy as np
import pandas as pd
from random import sample
from ecglib.data import EcgDataset
from torch.utils.data import DataLoader
from ecglib.preprocessing.composition import *
from sklearn.preprocessing import MultiLabelBinarizer

from .cifar_utils import (
    ImageDataset,
    UnsupervisedImageDataset,
    get_image_dataset_params,
)


def prepare_df_for_federated_training(
    cfg: dict,
    directories_key: str,
):
    df = read_dataframe_from_cfg(cfg, directories_key)
    if "ID_CLINIC" in df.columns:
        df["client"] = df["ID_CLINIC"]
        list_of_clients = get_list_of_clients(
            df,
            cfg.federated_params.min_sample_number,
            cfg.task_params.pathology_names,
            cfg.federated_params.amount_of_clients,
            cfg.federated_params.client_sample,
        )
        df["client"] = df["client"].apply(lambda x: np.where(list_of_clients == x)[0])
        df.reset_index(drop=True, inplace=True)
    print("Preprocess successfull\n")
    return df


def read_dataframe_from_cfg(
    cfg,
    directories_key="train_directories",
    mode="dataset",
):
    df = pd.DataFrame()
    for directories in cfg[mode]["data_sources"][directories_key]:
        df = pd.concat([df, pd.read_csv(directories, low_memory=False)])
    if "ecg_data" in cfg[mode].data_sources.train_directories[0]:
        df = ecg_dataframe_process(cfg, directories_key, ecg_info=df)
    if "semi_fed" not in cfg.federated_method._target_:
        df = df[df["target"] != -1]
    return df


def ecg_dataframe_process(
    cfg: dict,
    directories_key: str,
    ecg_info: pd.DataFrame,
    data_filtering: bool = True,
) -> tuple:
    """Filter ecg_info dataframe and create a targets

    :param cfg: Configuration
    :param directories_key: Key to extract directories for dataframe (train or test)
    :param ecg_info: dataframe with all map files
    :param data_filtering: Whether to do data filtering

    :return: ecg_info -- filtered dataframe
    """

    # Remove rows with filenames listed in filter_dataframe_files
    ecg_info = filter_by_another_dataframes(
        ecg_info, cfg["filter"]["data_sources"]["filter_dataframe_files"]
    )

    ecg_info = ecg_info[ecg_info[cfg.task_params.task_type].notna()]

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


def filter_by_another_dataframes(ecg_info, filter_dataframe_files):
    """
    filter the dataframe by name of files from others

    :param ecg_info: pandas Dataframe
    :param filter_dataframe_files: paths to files containing ecg filenames to be filtered

    :return: filtered Dataframe
    """
    for file_path in filter_dataframe_files:
        try:
            filter_df = pd.read_csv(file_path, low_memory=False)
        except pd.errors.EmptyDataError:
            continue
        filter_df_list = filter_df.file_name.to_list()
        ecg_info = ecg_info[~ecg_info.file_name.isin(filter_df_list)]
    return ecg_info


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

        existing_columns = set(v).intersection(set(df.columns))
        assert (
            len(existing_columns) != 0
        ), f"None of the specified pathologies {v} exist in the dataset."

        if existing_columns != set(v):
            print(
                f"Pathologies do not exist in the dataset: {set(v) - set(df.columns)}. Using only existing pathologies: {existing_columns}."
            )

        tmp = df[list(existing_columns)].apply(any, axis=1).astype(int)
        df.drop(columns=existing_columns, inplace=True)
        df[k] = tmp
    return df


def get_augmentation(cfg):
    augmentation_transform = cfg.ecg_record_params.augmentation.transforms
    if augmentation_transform:
        aug_list = []
        keys = augmentation_transform.keys()
        for _, key in enumerate(keys):
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


def get_list_of_clients(
    train_df, min_sample_number, pathology_names, client_amount, client_sample
):
    """
    get list of all clients (hospitals) with at least min_sample_number samples

    :param df: pandas Dataframe
    :param min_sample_number: min number of samples from one client

    :return: list
    """
    train_value_counts = train_df["client"].value_counts()

    # filter by min_sample_number
    available_clients = train_value_counts[
        train_value_counts >= min_sample_number
    ].index.tolist()

    if "ecg_data" in list(train_df["fpath"])[0]:
        # filter by pathology_names
        client_list = []
        num_clients = client_amount
        for id_clinic in available_clients:
            clinic_id_df = train_df[train_df["client"] == id_clinic]
            df_pathology_mask = clinic_id_df["scp_codes"].apply(
                lambda x: [item in ast.literal_eval(x) for item in pathology_names]
            )
            np_pathology_mask = np.array(df_pathology_mask.values.tolist())
            # check that for all path in pathology_names exists at least one positive sample
            entry_to_df = all(
                [any(np_pathology_mask[:, i]) for i in range(len(pathology_names))]
            )
            if entry_to_df:
                client_list.append(id_clinic)
                num_clients -= 1
            if num_clients == 0:
                break

        client_list = np.sort(
            np.array(client_list)[
                sample(
                    range(0, len(client_list)),
                    int(client_sample * len(client_list)),
                )
            ]
        )
        assert len(client_list) == client_amount, "min_sample_number should be less"
    else:
        client_list = np.array(available_clients[:client_amount])
    return client_list


def get_dataset_loader(
    df: pd.DataFrame,
    cfg,
    drop_last=True,
    mode="train",
    unsupervised=False,
    transforms=None,
):
    if "image_data" in cfg.dataset.data_sources.train_directories[0]:
        if unsupervised:
            dataset = UnsupervisedImageDataset(
                df=df,
                is_train=(mode == "train"),
                weak_transforms=transforms[mode]["weak"],
                strong_transforms=transforms[mode]["strong"],
            )
        else:
            image_size, mean, std = get_image_dataset_params(cfg, df)
            dataset = ImageDataset(df, mode, image_size, mean, std)
    else:
        augmentation = get_augmentation(cfg)
        ecg_target = df.target.values
        dataset = EcgDataset.for_train_from_config(
            df, ecg_target, augmentation, cfg, cfg.task_params.classes
        )
    loader = DataLoader(
        dataset,
        batch_size=cfg.training_params.batch_size,
        shuffle=(mode == "train"),
        num_workers=cfg.training_params.num_workers,
        drop_last=drop_last,
    )
    assert (
        len(loader) > 0
    ), f"len(dataloader) is 0, either lower the batch size, or put drop_last=False"
    return loader


def get_stratified_subsample(df, num_samples, random_state):
    """Create a subDataFrame with `num_samples` and stratified label distribution

    Args:
        df (pd.DataFrame): origin DataFrame
        num_samples (_type_): number of samples in subDataFrame

    return: sub_df (pd.DataFrame): sub DataFrame
    """
    sub_df = pd.DataFrame()
    # ecg have multilabel list of targets
    if isinstance(df["target"].iloc[0], list):
        ecg_type = True
        assert (
            len(df["target"].iloc[0]) == 1
        ), f"Methods with trust dataframe support only binary multilabel case, you provide: {df['target'].iloc[0]}"
        df.loc[:, "target"] = df["target"].apply(lambda x: x[0])
    # cifar case
    else:
        ecg_type = False
    for target in list(df.target.value_counts().keys()):
        tmp = df[df["target"] == target]
        weight = len(tmp) / len(df)
        amount = int(weight * num_samples)
        sub_df = pd.concat(
            [
                sub_df,
                tmp.sample(
                    n=amount,
                    random_state=random_state,
                ),
            ]
        )
    # Remove all rows from df, that are now in sub_df
    df = df[~df["fpath"].isin(list(sub_df["fpath"]))]
    # ECG case
    if ecg_type:
        df.loc[:, "target"] = df["target"].apply(lambda x: [x])
        sub_df.loc[:, "target"] = sub_df["target"].apply(lambda x: [x])
    return df, sub_df


def set_up_base_dir(cfg: dict):
    """Set up base directory to handle 'home'/'space' location

    Args:
        cfg (DictConfig): Programm configuration
    """
    # We have four types of directories: client train, server test, filter, server trust
    modes = ["dataset", "server_test", "filter", "trust_df"]
    for mode in modes:
        # Inside each type, we have different subtypes (train_directories/valid_directories)
        for type in cfg[mode]["data_sources"]:
            # Each subtype is a list of map_files
            for i in range(len(cfg[mode]["data_sources"][type])):
                path_without_base = "/".join(
                    cfg[mode]["data_sources"][type][i].split("/")[2:]
                )
                fpath = os.path.join("/", cfg.base_dir, path_without_base)
                cfg[mode]["data_sources"][type][i] = fpath
    return cfg
