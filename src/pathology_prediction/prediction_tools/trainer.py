import torch
import pandas as pd
from tqdm import tqdm

from torch.utils.data import DataLoader
from ecglib.data import EcgDataset

from utils.model_utils import get_model
from utils.data_utils import get_augmentation
from utils.metrics_utils import (
    metrics_report,
    select_best_validation_threshold,
    stopping_criterion,
)


class BaseTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.leads_num = len(cfg.ecg_record_params.leads)
        self.pathology_names = self._get_pathologies(
            cfg.task_params.pathology_names, cfg.task_params.merge_map
        )
        self.git_info = None
        self.augmentation = get_augmentation(cfg)
        self.train_loader = None
        self.valid_loader = None
        self.model = get_model(cfg, self.leads_num)
        self.criterion = None
        self.optimizer = self._init_optimizer(cfg.training_params.lr)
        self.scheduler = self._init_scheduler(
            cfg.training_params.factor, cfg.training_params.scheduler_patience
        )
        self.best_metrics = self._init_best_metrics(cfg.training_params.saving_metrics)
        self.epochs = cfg.training_params.epochs
        self.epochs_no_improve = 0
        self.prediction_threshold = cfg.training_params.prediction_threshold
        self.early_stopping_patience = cfg.training_params.early_stopping_patience
        self.tensor_logger = None
        self.device = "cpu"

    def _init_dataloaders(self, cfg, train_df, valid_df):
        train_loader = self.get_dataset_loader(
            train_df, cfg, self.augmentation, cfg.task_params.classes
        )
        valid_loader = self.get_dataset_loader(
            valid_df, cfg, self.augmentation, cfg.task_params.classes
        )
        return train_loader, valid_loader

    def _get_pathologies(self, pathology_names, pathologies_merge_map):
        if pathologies_merge_map:
            return list(pathologies_merge_map.keys())
        else:
            return pathology_names

    def _init_best_metrics(self, saving_metrics):
        return {metric: 1000 * (metric == "loss") for metric in saving_metrics}

    def _init_optimizer(self, learning_rate):
        return torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    def _init_scheduler(self, factor, scheduler_patience):
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=factor,
            patience=scheduler_patience,
            verbose=True,
        )

    def get_dataset_loader(
        self,
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

    def train(self):

        # ====================================== Train the model
        for ep in range(self.epochs):
            print("Epoch number:", ep)

            train_loss, val_loss, metrics = self.train_epoch()

            self.save_checkpoint(val_loss / len(self.valid_loader), metrics)

            print(
                f"\ntrain_loss:{train_loss / len(self.train_loader)}\n \
                \nval_loss: {val_loss / len(self.valid_loader)}\n \
                \nepochs no improve: {self.epochs_no_improve}\n"
            )

            if self.epochs_no_improve >= self.early_stopping_patience:
                print("Early stopping")
                break

    def train_epoch(self):
        train_loss = self.train_fn()
        val_loss, fin_targets, fin_outputs = self.eval_fn()
        try:
            metrics = self.calculate_metrics(fin_targets, fin_outputs)
        except:
            metrics = None
        self.scheduler.step(val_loss)

        return train_loss, val_loss, metrics

    def train_fn(self):
        sum_loss = 0
        self.model.train()

        for bi, batch in tqdm(
            enumerate(self.train_loader), total=len(self.train_loader)
        ):
            index, (input, targets) = batch

            inp = self.get_model_input(input)

            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inp)

            loss = self.get_loss_value(self.criterion, outputs, targets)

            loss.backward()
            sum_loss += loss.detach().item()

            self.optimizer.step()

        return sum_loss

    def eval_fn(self):
        self.model.eval()
        sum_loss = 0
        fin_targets = []
        fin_outputs = []

        with torch.no_grad():
            for bi, batch in tqdm(
                enumerate(self.valid_loader), total=len(self.valid_loader)
            ):
                index, (input, targets) = batch

                inp = self.get_model_input(input)
                targets = targets.to(self.device)

                outputs = self.model(inp)

                loss = self.get_loss_value(self.criterion, outputs, targets)

                sum_loss += loss.detach().item()

                fin_targets.extend(targets.tolist())
                fin_outputs.extend(outputs.tolist())

        return sum_loss, fin_targets, fin_outputs

    def get_model_input(self, input):
        ecg_signal = input[0]

        return ecg_signal.to(self.device)

    def get_loss_value(self, criterion, outputs, targets):
        return criterion(outputs, targets)

    def calculate_metrics(self, fin_targets, fin_outputs):
        sigmoid = torch.nn.Sigmoid()

        fin_outputs = sigmoid(torch.as_tensor(fin_outputs))
        self.prediction_threshold = select_best_validation_threshold(
            fin_targets, fin_outputs, self.cfg.training_params.metrics_threshold
        )
        results = (fin_outputs > self.prediction_threshold).float()
        metrics, _ = metrics_report(
            fin_targets, results.tolist(), self.pathology_names, fin_outputs
        )
        return metrics

    def save_checkpoint(self, val_loss, metrics):
        self.epochs_no_improve, self.best_metrics = stopping_criterion(
            val_loss, metrics, self.best_metrics, self.epochs_no_improve
        )
        if self.epochs_no_improve == 0:
            print("Model saved!")
            model_info = {
                "git": self.git_info,
                "model": self.model.state_dict(),
                "config_file": self.cfg,
            }
            checkpoint_dir = self.cfg.single_run_dir
            composed_name = "_".join([model["model_name"] for model in self.cfg.models])
            suffix_name = "_".join([pathology for pathology in self.pathology_names])
            checkpoint_path = f"{checkpoint_dir}/{self.leads_num}_leads_{composed_name}_{suffix_name}.pt"

            torch.save(model_info, checkpoint_path)
