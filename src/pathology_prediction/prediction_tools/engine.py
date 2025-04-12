import pandas as pd
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
import numpy as np
import random

from utils.data_utils import (
    get_list_of_clients,
    change_labels_of_clients,
    create_dataframe,
)
from utils.losses import get_loss
from trainer import BaseTrainer
from federated_trainer import FederatedTrainer

__all__ = [
    "select_train",
    "prepare_data_for_federated_learning",
]


def select_train(
    cfg: DictConfig,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame = None,
):
    if cfg.train_type == "train_all_data":
        trainer = BaseTrainer(cfg)
        if cfg.training_params.device == "cuda":
            trainer.device = "{}:{}".format(
                cfg.training_params.device, cfg.training_params.device_ids[0]
            )

        # if the method is chosen as just 1 client, who has the most amount of signals with the given pathology
        if cfg.federated_params.method == "biggest_client":

            train_df = pd.concat([train_df, valid_df], ignore_index=True)
            client_pathology_amount = dict()
            train_value_counts = train_df["ID_CLINIC"].value_counts()
            unique_train_values_with_min_occurrences = train_value_counts[
                train_value_counts > cfg.federated_params.min_sample_number
            ].index.tolist()
            for client in unique_train_values_with_min_occurrences:
                df = train_df[train_df["ID_CLINIC"] == client]
                client_pathology_amount[client] = sum(
                    [x for xs in df["target"].values for x in xs]
                )

            client = max(client_pathology_amount, key=client_pathology_amount.get)
            train_df = train_df[train_df["ID_CLINIC"] == client]

            train_df, valid_df = train_test_split(
                train_df,
                test_size=0.2,
                stratify=train_df["target"],
                random_state=42,
            )

        trainer.train_loader, trainer.valid_loader = trainer._init_dataloaders(
            cfg, train_df, valid_df
        )
        trainer.criterion = get_loss(
            loss_cfg=cfg.loss,
            device=trainer.device,
            df=train_df,
            loaders=[trainer.train_loader, trainer.valid_loader],
        )
        trainer.train()
    elif cfg.train_type == "federated_training":
        federated_trainer = FederatedTrainer(cfg)
        print("Getting list of clients:")
        train_df = pd.concat([train_df, valid_df], ignore_index=True)
        federated_trainer.list_of_clients = get_list_of_clients(
            train_df,
            cfg.federated_params.min_sample_number,
            cfg.task_params.pathology_names,
            cfg.federated_params.amount_of_clients,
            cfg.federated_params.client_sample,
        )
        if cfg.training_params.device == "cuda":
            federated_trainer.device = "{}:{}".format(
                cfg.training_params.device, cfg.training_params.device_ids[0]
            )
        if federated_trainer.attack:
            if federated_trainer.attacking_method == "label_flipping":
                np.random.seed(cfg.federated_params.random_state)
                federated_trainer.attacking_clients = np.random.choice(
                    range(len(federated_trainer.list_of_clients)),
                    size=int(
                        len(federated_trainer.list_of_clients)
                        * cfg.federated_params.amount_of_attackers
                    ),
                    replace=False,
                )
                print(
                    f"Attacking client indeces: {federated_trainer.attacking_clients}\nAttacking method: {federated_trainer.attacking_method}"
                )
                train_df = change_labels_of_clients(
                    train_df,
                    federated_trainer.attacking_method,
                    federated_trainer.list_of_clients,
                    federated_trainer.attacking_clients,
                    cfg.federated_params.percent_of_changed_labels,
                )
                print("Succesfully flipped the labels")
            elif federated_trainer.attacking_method == "random_rounds_random_clients":
                federated_trainer.attacking_rounds = random.sample(
                    list(range(federated_trainer.number_com_rounds)),
                    int(
                        federated_trainer.number_com_rounds
                        * cfg.federated_params.amount_of_attack_rounds
                    ),
                )
                print(f"Attacking rounds: {federated_trainer.attacking_rounds}")
        (
            federated_trainer.train_df,
            federated_trainer.valid_df,
        ) = prepare_data_for_federated_learning(
            train_df,
            federated_trainer.list_of_clients,
            cfg.federated_params.random_state,
        )

        if "FLTrust" in cfg.federated_params.method:
            federated_trainer.fltrust_train_df = create_dataframe(
                cfg, "train_directories", dataset=cfg.federated_params.fltrust_dataset
            )
            federated_trainer.fltrust_valid_df = create_dataframe(
                cfg, "valid_directories", dataset=cfg.federated_params.fltrust_dataset
            )
            print("Initialized FLTrust dataset")

        federated_trainer.test_df = create_dataframe(
            cfg, "ptbxl_test_data", dataset="test_ptbxl"
        )
        print(f"Number of clients: {len(federated_trainer.list_of_clients)}")

        federated_trainer.federated_training()


def prepare_data_for_federated_learning(train_df, list_of_clients, random_state):
    train_df = train_df[train_df["ID_CLINIC"].isin(list_of_clients)]
    train_df["stratify_column"] = (
        train_df["target"].astype(str) + "_" + train_df["ID_CLINIC"].astype(str)
    )

    # Split the data into train and validation sets, ensuring stratification

    # TODO stratifying

    train_df, valid_df = train_test_split(
        train_df,
        test_size=0.2,
        stratify=train_df["ID_CLINIC"],
        random_state=random_state,
    )

    train_df.drop(columns=["stratify_column"], inplace=True)
    valid_df.drop(columns=["stratify_column"], inplace=True)

    return train_df, valid_df
