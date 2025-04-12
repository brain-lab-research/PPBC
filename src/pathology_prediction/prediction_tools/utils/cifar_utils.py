import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms

from utils.data_utils import change_labels_of_clients


def read_cifar_dataset(
    cfg: dict,
):

    print("Reading datasets...", flush=True)
    if cfg.cifar.dataset == "10_clients":
        df = pd.read_csv(
            f"/{cfg.cifar.base_dir}/image_data/cifar/10_clients/10_clients_train_map_file.csv"
        )
        list_of_clients = list(
            range(1, min(11, cfg.federated_params.amount_of_clients + 1))
        )
    elif cfg.cifar.dataset == "100_clients":
        df = pd.read_csv(
            f"/{cfg.cifar.base_dir}/image_data/cifar/100_clients/100_clients_train_map_file.csv"
        )
        list_of_clients = list(
            range(1, min(101, cfg.federated_params.amount_of_clients + 1))
        )

    if cfg.federated_params.attack:
        print(
            f"\nInitializing an attack...",
            flush=True,
        )
        if cfg.federated_params.attacking_method == "label_flipping":
            np.random.seed(cfg.federated_params.random_state)
            attacking_clients = np.random.choice(
                range(len(list_of_clients)),
                size=int(
                    len(list_of_clients) * cfg.federated_params.amount_of_attackers
                ),
                replace=False,
            )
            print(
                f"Attacking client indeces: {[x + 1 for x in attacking_clients]}\nAttacking method: label_flipping",
                flush=True,
            )
            df = change_labels_of_clients(
                df,
                cfg.federated_params.attacking_method,
                list_of_clients,
                attacking_clients,
                cfg.federated_params.percent_of_changed_labels,
                "cifar",
            )
            print(f"Succesfully flipped the labels", flush=True)

    df = df[df["target"].notna()]
    print("Preprocess successfull", flush=True)

    df.reset_index(drop=True, inplace=True)
    return df


class cifar_dataset(Dataset):
    def __init__(self, df, transform):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        image = Image.open(self.df["fpath"][index])
        image = self.transform(image)
        label = self.df["target"][index]
        return index, ([image], label)


def init_cifar_loader(df, transform_mode, batch_size, shuffle=True):
    if transform_mode == "train":
        transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    elif transform_mode == "valid":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    dataset = cifar_dataset(df, transform)
    loader = DataLoader(dataset, batch_size, shuffle=shuffle)
    return loader


def calculate_cifar_metrics(fin_targets, fin_outputs):
    softmax = torch.nn.Softmax(dim=1)
    soft_fin_outputs = softmax(torch.as_tensor(fin_outputs))
    fin_outputs_indices = soft_fin_outputs.max(dim=1)[1]
    print(
        f"\nAccuracy: {accuracy_score(torch.as_tensor(fin_targets), fin_outputs_indices)}"
    )
