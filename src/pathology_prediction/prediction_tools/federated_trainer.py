import copy
import torch
import numpy as np
from collections import OrderedDict

from trainer import BaseTrainer
from utils.model_utils import get_model
from utils.losses import get_loss
from tqdm import tqdm

from torch.nn.functional import relu

from utils.metrics_utils import select_best_validation_threshold
from utils.data_utils import change_labels_of_clients


class FederatedTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.federated_method = cfg.federated_params.method
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        self.current_com_round = 0
        self.number_com_rounds = cfg.federated_params.communication_rounds
        self.list_of_clients = None
        self.global_model = get_model(cfg, self.leads_num)
        self.model = None
        self.best_global_train_loss = 10000
        self.best_global_val_loss = 10000
        self.lr = self.cfg.training_params.lr
        self.best_round = 0
        self.rounds_no_improve = 0
        self.states = []
        self.attack = cfg.federated_params.attack
        if self.attack:
            self.attacking_clients = []
            self.attacking_method = cfg.federated_params.attacking_method
            if self.attacking_method == "random_rounds_random_clients":
                self.attacking_rounds = []
        if "FLTrust" in self.federated_method:
            self.fltrust_train_df = None
            self.fltrust_valid_df = None
            if self.federated_method == "FLTrust_new":
                self.fltrust_threshold = None
                self.sigmoid = torch.nn.Sigmoid()

    def init_client_dalaloaders(self, client_idx):
        client_train_df = self.train_df[
            self.train_df["ID_CLINIC"] == self.list_of_clients[client_idx]
        ]
        client_valid_df = self.valid_df[
            self.valid_df["ID_CLINIC"] == self.list_of_clients[client_idx]
        ]
        if (
            self.attack
            and self.attacking_method == "random_rounds_random_clients"
            and client_idx in self.attacking_clients
        ):
            client_train_df = change_labels_of_clients(
                client_train_df,
                self.attacking_method,
                p=self.cfg.federated_params.percent_of_changed_labels
            )
            client_valid_df = change_labels_of_clients(
                client_valid_df,
                self.attacking_method,
                p=self.cfg.federated_params.percent_of_changed_labels
            )
        print(
            f"Number of training samples of client {self.list_of_clients[client_idx]}: {len(client_train_df)}"
        )
        print(
            f"Number of valid samples of client {self.list_of_clients[client_idx]}: {len(client_valid_df)}"
        )
        print(client_train_df["target"].value_counts())
        print(client_valid_df["target"].value_counts())
        self.train_loader = self.get_dataset_loader(
            client_train_df, self.cfg, self.augmentation, self.cfg.task_params.classes
        )
        self.valid_loader = self.get_dataset_loader(
            client_valid_df, self.cfg, self.augmentation, self.cfg.task_params.classes
        )

    def _init_client_round(self):
        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=self.train_df,
            loaders=[self.train_loader, self.valid_loader],
        )

        # storing best model (by val_loss)

        best_client_train_loss = 0
        best_client_val_loss = 0

        # ====================================== Select training parameters

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            factor=self.cfg.training_params.factor,
            patience=self.cfg.training_params.scheduler_patience,
            verbose=True,
        )
        self.best_metrics = {
            metric: 1000 * (metric == "loss")
            for metric in self.cfg.training_params.saving_metrics
        }
        self.epochs_no_improve = 0

        return best_client_train_loss, best_client_val_loss

    def _init_controls(self):
        self.server_control = {}
        for k, v in self.global_model.state_dict().items():
            self.server_control[k] = torch.zeros_like(v.data)
        self.clients_control = [self.server_control] * len(self.list_of_clients)
        self.updated_clients_control = self.clients_control

    def federated_training(self):
        if self.federated_method == "SCAFFOLD":
            self._init_controls()

        for com_round in range(self.number_com_rounds):
            print(f"Communication round {com_round}")
            self.current_com_round = com_round
            if (
                self.attack
                and self.attacking_method == "random_rounds_random_clients"
                and self.current_com_round in self.attacking_rounds
            ):
                self.attacking_clients = np.random.choice(
                    range(len(self.list_of_clients)),
                    size=int(
                        len(self.list_of_clients)
                        * self.cfg.federated_params.amount_of_attackers
                    ),
                    replace=False,
                )
                print(f"Attack this round by these clients: {self.attacking_clients}")

            updated_list_of_trained_model_parameters = []
            global_train_loss = 0
            global_val_loss = 0

            for client_idx in range(len(self.list_of_clients)):

                if self.attack:
                    print(f"Client attacking: {client_idx in self.attacking_clients}")
                # Filter train_df by ID_CLINIC
                self.init_client_dalaloaders(client_idx)

                # Perform federated learning round
                self.model = copy.deepcopy(self.global_model)

                if self.federated_method == "SCAFFOLD":
                    self.client_idx = client_idx

                client_model_weights, client_train_loss, client_val_loss = (
                    self.federated_training_round()
                )

                updated_list_of_trained_model_parameters.append(client_model_weights)
                global_train_loss += client_train_loss * (
                    len(self.train_loader) / len(self.train_df)
                )
                global_val_loss += client_val_loss * (
                    len(self.valid_loader) / len(self.valid_df)
                )

            if "FLTrust" in self.federated_method:

                print("\nStarted training server side model\n")

                self.train_loader = self.get_dataset_loader(
                    self.fltrust_train_df,
                    self.cfg,
                    self.augmentation,
                    self.cfg.task_params.classes,
                )
                self.valid_loader = self.get_dataset_loader(
                    self.fltrust_valid_df,
                    self.cfg,
                    self.augmentation,
                    self.cfg.task_params.classes,
                )

                self.model = copy.deepcopy(self.global_model)

                server_model_weights, _, _ = self.federated_training_round()

                if self.federated_method == "FLTrust_new":
                    self.valid_loader = self.get_dataset_loader(
                        self.test_df,
                        self.cfg,
                        self.augmentation,
                        self.cfg.task_params.classes,
                    )
                    _, fin_targets, fin_outputs = self.eval_fn()
                    fin_outputs = self.sigmoid(torch.as_tensor(fin_outputs))
                    self.fltrust_threshold = select_best_validation_threshold(
                        fin_targets,
                        fin_outputs,
                        self.cfg.training_params.metrics_threshold,
                    )

                updated_list_of_trained_model_parameters.append(server_model_weights)

                print("\nTrained server side model\n")

                self.attacking_clients = []

            self.save_round(
                global_train_loss,
                global_val_loss,
                com_round,
                updated_list_of_trained_model_parameters,
            )

    def federated_training_round(self):

        # ====================================== Select criterion

        best_client_train_loss, best_client_val_loss = self._init_client_round()

        # ====================================== Train the model
        for ep in range(self.cfg.federated_params.round_epochs):
            print("Epoch number:", ep)

            train_loss, val_loss, metrics = self.train_epoch()

            # self.save_checkpoint(val_loss / len(self.valid_loader), metrics)

            print(
                f"\ntrain_loss:{train_loss  / len(self.train_loader)}\n \
                    \nval_loss: {val_loss  / len(self.valid_loader)}\n"
            )

            if self.epochs_no_improve == 0:
                best_client_train_loss = train_loss
                best_client_val_loss = val_loss
            if self.epochs_no_improve >= self.early_stopping_patience:
                print("Early stopping")
                break

        best_client_model_weights = OrderedDict()
        for key, weights in self.states[0].items():
            best_client_model_weights[key] = 0
        for state in self.states:
            for key, weights in state.items():
                best_client_model_weights[key] = (
                    best_client_model_weights[key] + weights
                )

        if self.federated_method == "SCAFFOLD":
            self._update_client_control(best_client_model_weights)

        self.states = []

        return best_client_model_weights, best_client_train_loss, best_client_val_loss

    def train_fn(self):
        sum_loss = 0
        old_state = copy.deepcopy(self.model)
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

            if self.federated_method == "SCAFFOLD":
                self._add_control()

        self.model.eval()
        state = OrderedDict()
        for (key, weights1), (_, weights2) in zip(
            self.model.state_dict().items(), old_state.state_dict().items()
        ):
            state[key] = weights1 - weights2

        print(f"\nState saved!")
        self.states.append(state)
        self.model.train()

        return sum_loss

    def save_round(
        self,
        global_train_loss,
        global_val_loss,
        com_round,
        updated_list_of_trained_model_parameters,
    ):
        if global_val_loss < self.best_global_val_loss:
            self.best_global_val_loss = global_val_loss
            self.best_global_train_loss = global_train_loss
            self.best_round = com_round
            self.rounds_no_improve = 0
        else:
            self.rounds_no_improve += 1

        print("Current_global_loss:")
        print(
            f"Global train loss: {global_train_loss}\n Global val loss: {global_val_loss}"
        )

        if self.rounds_no_improve == self.cfg.federated_params.rounds_no_improve:
            self.rounds_no_improve = 0
            self.lr *= self.cfg.federated_params.lr_scheduler_factor

        self.update_global_model(updated_list_of_trained_model_parameters, com_round)

        try:
            self.valid_loader = self.get_dataset_loader(
                self.test_df, self.cfg, self.augmentation, self.cfg.task_params.classes
            )
            self.model = copy.deepcopy(self.global_model)
            sum_loss, fin_targets, fin_outputs = self.eval_fn()
            self.calculate_metrics(fin_targets, fin_outputs)
            print(f"Round number: {com_round}\nTest loss: {sum_loss}")
        except:
            pass

        print("Current_best_metrics")
        print(
            f"Best round: {self.best_round}\nTrain loss: {self.best_global_train_loss}\n Val_loss: {self.best_global_val_loss}"
        )

    def update_global_model(self, updated_list_of_trained_model_parameters, com_round):
        # TO DO: add client sampling

        averaged_model_weights = self.federated_step(
            updated_list_of_trained_model_parameters
        )

        # Update model weights and global model
        self.global_model = get_model(self.cfg, self.leads_num)
        self.global_model.load_state_dict(averaged_model_weights)

        checkpoint_path = f"{self.cfg.single_run_dir}/{self.leads_num}_leads_federated_learning_round_{com_round}.pt"
        torch.save(averaged_model_weights, checkpoint_path)

    # FedAvg: https://arxiv.org/abs/1602.05629
    def federated_step(self, updated_list_of_trained_model_parameters):
        num_clients = len(self.list_of_clients)

        # Initialize the model
        aggregated_weights = self.global_model.state_dict()

        if self.federated_method == "FLTrust":
            client_directions = []
            trust_scores = []
            server_direction = torch.cat(
                [
                    x.flatten()
                    for x in list(updated_list_of_trained_model_parameters[-1].values())
                ]
            )
            for i in range(num_clients):
                client_directions.append(
                    torch.cat(
                        [
                            x.flatten()
                            for x in list(
                                updated_list_of_trained_model_parameters[i].values()
                            )
                        ]
                    )
                )
                trust_scores.append(
                    relu(
                        torch.dot(server_direction, client_directions[i])
                        / (
                            torch.norm(server_direction)
                            * torch.norm(client_directions[i])
                        )
                    )
                )
                print(f"Client {i} trust score: {trust_scores[i]}")
            for i in range(num_clients):
                modified_client_model_weights = OrderedDict()
                for key, weights in updated_list_of_trained_model_parameters[i].items():
                    modified_client_model_weights[key] = (
                        (1 / torch.stack(trust_scores, dim=0).sum(dim=0))
                        * trust_scores[i]
                        * (
                            torch.norm(server_direction)
                            / torch.norm(client_directions[i])
                        )
                        * weights
                    )
                updated_list_of_trained_model_parameters[i] = (
                    modified_client_model_weights
                )

        elif self.federated_method == "FLTrust_new":
            signal = self.test_df[
                self.test_df["target"].apply(lambda x: 0 in x)
            ].sample(
                n=self.cfg.federated_params.fltrust_new_sample_amount,
                random_state=self.cfg.federated_params.random_state,
            )
            self.valid_loader = self.get_dataset_loader(
                signal,
                self.cfg,
                self.augmentation,
                self.cfg.task_params.classes,
                drop_last=False,
            )
            _, _, fin_outputs = self.eval_fn()
            server_result = self.sigmoid(torch.as_tensor(fin_outputs))
            trust_scores = []
            for i in range(num_clients):

                tmp_weights = self.global_model.state_dict()
                for key, weights in updated_list_of_trained_model_parameters[i].items():
                    tmp_weights[key] = tmp_weights[key] + weights

                self.model = get_model(self.cfg, self.leads_num)
                self.model.load_state_dict(tmp_weights)
                _, _, fin_outputs = self.eval_fn()
                client_result = self.sigmoid(torch.as_tensor(fin_outputs))
                trust_score = []
                for j in range(self.cfg.federated_params.fltrust_new_sample_amount):
                    trust_score.append(
                        self.count_trust_score_for_new_fltrust(
                            server_result[j], client_result[j]
                        )
                    )
                trust_scores.append(
                    (
                        sum(trust_score)
                        / self.cfg.federated_params.fltrust_new_sample_amount
                    )
                )
                print(f"Client {i} trust score: {trust_scores[i]}")
            trust_scores = torch.Tensor(trust_scores).to(self.device)
            for i in range(num_clients):
                modified_client_model_weights = OrderedDict()
                for key, weights in updated_list_of_trained_model_parameters[i].items():
                    modified_client_model_weights[key] = (
                        (1 / sum(trust_scores)) * trust_scores[i] * weights
                    )
                updated_list_of_trained_model_parameters[i] = (
                    modified_client_model_weights
                )

        # Going througth all the clients, count client_weight
        # and add weights to the aggregated model
        for i in range(num_clients):
            # filter by ID_CLINIC and count weight based on data size
            # client_train_df = self.train_df[self.train_df["ID_CLINIC"] == self.list_of_clients[i]]
            # client_weight = len(client_train_df) / len(self.train_df)

            # Count weights
            for key, weights in updated_list_of_trained_model_parameters[i].items():
                aggregated_weights[key] = aggregated_weights[key] + weights * (
                    1 / num_clients
                )

        if self.federated_method == "SCAFFOLD":
            self._update_server_control()

        return aggregated_weights

    def get_loss_value(self, criterion, outputs, targets):
        loss = criterion(outputs, targets)
        if (
            self.federated_method == "FedProx"
            and self.current_com_round >= self.cfg.federated_params.num_fedavg_rounds
        ):
            proximity_loss = self._count_proximity()
            loss += proximity_loss
        return loss

    def _count_proximity(self):
        return (
            0.5
            * self.cfg.federated_params.fed_prox_lambda
            * sum(
                [
                    (p.float() - q.float()).norm() ** 2
                    for (_, p), (_, q) in zip(
                        self.model.state_dict().items(),
                        self.global_model.state_dict().items(),
                    )
                ]
            )
        )

    def _add_control(self):
        with torch.no_grad():
            for k, v in self.global_model.named_parameters():
                if v.requires_grad:
                    v.add_(
                        self.lr
                        * (
                            self.server_control[k].data
                            - self.clients_control[self.client_idx][k].data
                        )
                    )

    def _update_client_control(self, state):
        # Option II in (4) from original paper https://arxiv.org/pdf/1910.06378.pdf
        coef = -1 / (self.cfg.federated_params.round_epochs * self.lr)
        for k, v in state.items():
            self.updated_clients_control[self.client_idx][k] = (
                self.clients_control[self.client_idx][k]
                - self.server_control[k]
                + coef * v
            )

    def _update_server_control(self):
        # update c global
        for k in self.server_control.keys():
            add_factor = sum(
                c_plus[k] - c[k]
                for c_plus, c in zip(self.updated_clients_control, self.clients_control)
            )
            self.server_control[k] += 1 / len(self.list_of_clients) * add_factor

        # reinit client c as updated for new round
        self.clients_control = self.updated_clients_control

    def _state_dict_diff(self, model1_state_dict, model2_state_dict):
        diff = OrderedDict()
        for (key, weights1), (_, weights2) in zip(
            model1_state_dict.items(), model2_state_dict.items()
        ):
            diff[key] = weights1 - weights2
        return diff

    def count_trust_score_for_new_fltrust(self, server_result, client_result):
        if (client_result - self.fltrust_threshold) * (
            server_result - self.fltrust_threshold
        ) < 0:
            return 0
        else:
            return 2 * (
                1 - self.sigmoid(torch.exp(10 * abs(client_result - server_result)) - 1)
            )
