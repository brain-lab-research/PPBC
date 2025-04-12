import time
import copy
import torch
import numpy as np
import pandas as pd
from collections import OrderedDict
from torch.nn.functional import relu, softmax
from sklearn.model_selection import train_test_split
from hydra.utils import instantiate

import sys

sys.path.append("..")

from utils.losses import get_loss
from utils.model_utils import get_model
from utils.cifar_utils import init_cifar_loader, calculate_cifar_metrics
from utils.data_utils import (
    create_dataframe,
    get_list_of_clients,
    change_labels_of_clients,
    get_dataset_loader,
)
from utils.optimizers_utils import FedAdam
from trainer import BaseTrainer

sys.path.pop(0)


class Client(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.federated_method = cfg.federated_params.method
        self.rank = None
        self.train_df = None
        self.aggregated_state = OrderedDict()
        self.current_com_round = 0
        self.global_model_state = None
        self.attacking = False
        self.optimizer = instantiate(cfg.optimizer, params=self.model.parameters())
        self.opt_name = self._get_opt_name()
        if self.opt_name == "FedAdam":
            self.v_t = {
                k: torch.zeros_like(v) for k, v in self.model.named_parameters()
            }
            param_groups = [
                {"params": p, "name": name} for name, p in self.model.named_parameters()
            ]
            self.optimizer = FedAdam(
                params=param_groups,
                lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay,
                betas=cfg.optimizer.betas,
                eps=cfg.optimizer.eps,
                vt_momentum=cfg.optimizer.vt_momentum,
            )
            self.init_vt = True
            assert (
                not cfg.federated_params.opt_from_train
            ), "FedAdam suppose optimizers communication with Serevr. Initialization from train is not allowed"
        if (
            cfg.federated_params.attack
            and cfg.federated_params.attacking_method == "random_rounds_random_clients"
        ):
            self.attacked_train_df = None
        if cfg.federated_params.method == "RECESS":
            assert self.opt_name == "SGD", "RECESS strategy support only sgd optimizer"
            self.abnormality_alpha = 0
        if cfg.federated_params.attack:
            if cfg.federated_params.attacking_method == "delayed_grad":
                self.delayed_grad = OrderedDict()
                self.delayed_grad_queue = []
                self.delayed_rounds = cfg.federated_params.num_delayed_rounds


    def init_loaders(self):
        self.train_df, valid_df = train_test_split(
            self.train_df,
            test_size=0.2,
            stratify=self.train_df["target"],
            random_state=self.cfg.federated_params.random_state,
        )
        self.train_loader = choose_correct_loader(
            self.cfg,
            self.train_df,
            "train",
        )
        self.valid_loader = choose_correct_loader(
            self.cfg,
            valid_df,
            "valid",
        )

    def train(self, mode="train"):
        if self.cfg.federated_params.opt_from_train:
            self.optimizer = instantiate(
                self.cfg.optimizer, params=self.model.parameters()
            )
        # Updade global Adam second momentum after first round
        if self.opt_name == "FedAdam" and not self.init_vt:
            self.optimizer.update_v_t(self.v_t)

        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=self.train_df,
            loaders=[self.train_loader, self.valid_loader],
        )
        if (
            self.attacking
            and self.cfg.federated_params.attacking_method
            == "random_rounds_random_clients"
        ):
            self.train_loader = choose_correct_loader(
                self.cfg,
                self.attacked_train_df,
                "train",
            )
            self.criterion = get_loss(
                loss_cfg=self.cfg.loss,
                device=self.device,
                df=self.attacked_train_df,
                loaders=[self.train_loader, self.valid_loader],
            )

        states = []
        start = time.time()
        self.global_model_state = copy.deepcopy(self.model).state_dict()

        for _ in range(self.cfg.federated_params.round_epochs):
            old_state = copy.deepcopy(self.model)
            state = OrderedDict()
            self.model.train()
            if self.opt_name == "FedAdam":
                assert (
                    self.cfg.federated_params.round_epochs == 1
                ), "FedAdam support only 1 local epoch at the current moment"
                total_gradients = OrderedDict()
                num_iters = 0

            for batch in self.train_loader:
                index, (input, targets) = batch

                inp = self.get_model_input(input)
                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inp)

                loss = self.get_loss_value(self.criterion, outputs, targets)

                loss.backward()

                # Sum up gradients on iteration
                if self.opt_name == "FedAdam":
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                if name in total_gradients:
                                    total_gradients[name] += param.grad
                                else:
                                    total_gradients[name] = param.grad.clone()
                    num_iters += 1

                self.optimizer.step()

            self.model.eval()

            if (
                self.attacking
                and self.cfg.federated_params.attacking_method == "sign_flip"
            ):
                for (key, weights1), (_, weights2) in zip(
                    self.model.state_dict().items(), old_state.state_dict().items()
                ):
                    if "bn" not in key and "running" not in key:
                        state[key] = weights2 - weights1
                    else:
                        state[key] = weights1 - weights2
            else:
                for (key, weights1), (_, weights2) in zip(
                    self.model.state_dict().items(), old_state.state_dict().items()
                ):
                    state[key] = weights1 - weights2

            states.append(state)

            # Collect v_t from optimizer to gather to server on first iteration
            if self.opt_name == "FedAdam":
                if self.init_vt:
                    self.v_t = self.optimizer.get_v_t()

            # averaging client gradients
            if self.opt_name == "FedAdam":
                for name in total_gradients.keys():
                    total_gradients[name] /= num_iters
                self.aggregated_gradients = total_gradients
                self.init_vt = False
            

        end = time.time()

        if (
            self.attacking
            and self.cfg.federated_params.attacking_method == "random_grad"
        ):
            random_weights = get_model(self.cfg, len(self.cfg.ecg_record_params.leads))
            for (key, weights1), (_, weights2) in zip(
                self.global_model_state.items(), random_weights.state_dict().items()
            ):
                if "bn" not in key and "running" not in key:
                    self.aggregated_state[key] = weights2 - weights1
                else:
                    self.aggregated_state[key] = self.model.state_dict()[key] - weights1
        elif (
            self.cfg.federated_params.attack
            and self.attacking
            and self.cfg.federated_params.attacking_method == "delayed_grad"
        ):
            if self.current_com_round in range(0, self.delayed_rounds):
                random_weights = get_model(self.cfg, len(self.cfg.ecg_record_params.leads))
                for (key, weights1), (_, weights2) in zip(
                    self.global_model_state.items(), random_weights.state_dict().items()
                ):
                    if "bn" not in key and "running" not in key:
                        self.aggregated_state[key] = weights2 - weights1
                    else:
                        self.aggregated_state[key] = self.model.state_dict()[key] - weights1    
            else:    
                self.aggregated_state = copy.deepcopy(self.delayed_grad_queue[0])
                self.delayed_grad_queue.pop(0)

            for key, weights in states[0].items():
                    self.delayed_grad[key] = 0
            for state in states:
                for key, weights in state.items():
                    self.delayed_grad[key] = self.delayed_grad[key] + weights
                    if "bn" in key or "running" in key:
                        self.aggregated_state[key] = self.delayed_grad[key]
            self.delayed_grad_queue.append(self.delayed_grad)
                
        else:
            for key, weights in states[0].items():
                self.aggregated_state[key] = 0
            for state in states:
                for key, weights in state.items():
                    self.aggregated_state[key] = self.aggregated_state[key] + weights

        if self.attacking:
            self.train_loader = choose_correct_loader(
                self.cfg,
                self.train_df,
                "train",
            )
            attacked_state = OrderedDict()
            for key, weights in self.global_model_state.items():
                attacked_state[key] = weights + self.aggregated_state[key]
            self.model.load_state_dict(attacked_state)

        if mode == "train":
            print(
                f"Client {self.rank} finished training in {end - start} seconds, state is saved",
                flush=True,
            )

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
                        self.global_model_state.items(),
                    )
                ]
            )
        )

    def print_metrics(self):
        print(f"\nClient {self.rank} results:", flush=True)
        _, fin_targets, fin_outputs = self.eval_fn()
        if self.cfg.dataset == "cifar":
            calculate_cifar_metrics(fin_targets, fin_outputs)
        else:
            self.calculate_metrics(fin_targets, fin_outputs)
        norm = torch.norm(
            torch.cat([x.flatten() for x in list(self.aggregated_state.values())])
        )
        print(f"\nClient {self.rank} gradient norm: {norm}", flush=True)

    def _get_opt_name(self):
        return type(self.optimizer).__name__

    def recess_detection(self):

        modified_gradient = OrderedDict()
        new_state = OrderedDict()
        cur_norm = torch.linalg.norm(
            torch.cat([x.flatten() for x in list(self.aggregated_state.values())])
        )
        for key, weights in self.aggregated_state.items():
            modified_gradient[key] = weights / cur_norm
            new_state[key] = self.global_model_state[key] + modified_gradient[key]
        self.model.load_state_dict(new_state)

        self.train(mode="recess")

        old_grad = torch.cat([x.flatten() for x in list(modified_gradient.values())])
        new_grad = torch.cat(
            [x.flatten() for x in list(self.aggregated_state.values())]
        )
        cos_sim = torch.dot(old_grad, new_grad) / (
            torch.linalg.norm(old_grad) * torch.linalg.norm(new_grad)
        )
        self.abnormality_alpha = -cos_sim / torch.linalg.norm(new_grad)


class Server(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.federated_method = cfg.federated_params.method
        self.aggregated_state = OrderedDict()
        self.states = [None for _ in range(cfg.federated_params.amount_of_clients + 1)]
        self.optimizer = instantiate(cfg.optimizer, params=self.model.parameters())
        self.opt_name = self._get_opt_name()
        if self.opt_name == "FedAdam":
            self.aggregated_gradients = OrderedDict()
            self.gradients = [
                None for _ in range(cfg.federated_params.amount_of_clients + 1)
            ]
            self.v_t = {
                k: torch.zeros_like(v) for k, v in self.model.named_parameters()
            }
            self.v_ts = [
                None for _ in range(cfg.federated_params.amount_of_clients + 1)
            ]
            param_groups = [
                {"params": p, "name": name} for name, p in self.model.named_parameters()
            ]
            self.optimizer = FedAdam(
                params=param_groups,
                lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay,
                betas=cfg.optimizer.betas,
                eps=cfg.optimizer.eps,
                vt_momentum=cfg.optimizer.vt_momentum,
            )
        self.test_loss = 0
        self.best_round = (0, 0)
        self.test_df = None
        if "FLTrust" in self.federated_method:
            self.fltrust_train_df = None
            self.global_model_state = None
            if self.federated_method == "FLTrust_new":
                self.ts_test_df = None
                if self.cfg.dataset == "cifar":
                    self.activation = torch.nn.Softmax(dim=1)
                else:
                    self.activation = torch.nn.Sigmoid()
        if "TS_momentum" in self.federated_method:
            self.ts_test_df = None
            self.prev_trust_scores = [
                1 / cfg.federated_params.amount_of_clients
            ] * cfg.federated_params.amount_of_clients
            self.momentum_beta = cfg.federated_params.momentum_beta
        if self.federated_method == "RECESS":
            self.abnormality_alpha = 0
            self.abnormality_alphas = [
                0 for _ in range(cfg.federated_params.amount_of_clients + 1)
            ]
            self.prev_trust_scores = [1] * cfg.federated_params.amount_of_clients
            self.baseline_decreased_score = (
                cfg.federated_params.baseline_decreased_score
            )
            self.init_trust_score = cfg.federated_params.init_trust_score
        if cfg.federated_params.attack and cfg.federated_params.attacking_method=="IPM":
            self.attacking_clients = [
                None for _ in range(int(cfg.federated_params.amount_of_clients*cfg.federated_params.amount_of_attackers))
            ]
            self.ipm_eps = cfg.federated_params.ipm_eps

    def init_loaders(self):
        self.valid_loader = choose_correct_loader(
            self.cfg,
            self.test_df,
            "valid",
        )
        if "FLTrust" in self.federated_method:
            self.train_loader = choose_correct_loader(
                self.cfg,
                self.fltrust_train_df,
                "train",
            )
        if (
            "FLTrust_new" in self.federated_method
            or self.federated_method == "TS_momentum"
        ):
            self.ts_test_df = pd.DataFrame()
            if self.cfg.dataset != "cifar":
                self.test_df["target"] = self.test_df["target"].apply(lambda x: x[0])
            for target in list(self.test_df.target.value_counts().keys()):
                tmp = self.test_df[self.test_df["target"] == target]
                weight = len(tmp) / len(self.test_df)
                amount = int(
                    weight * self.cfg.federated_params.fltrust_new_sample_amount
                )
                self.ts_test_df = pd.concat(
                    [
                        self.ts_test_df,
                        tmp.sample(
                            n=amount,
                            random_state=self.cfg.federated_params.random_state,
                        ),
                    ]
                )
            if self.cfg.dataset != "cifar":
                self.test_df["target"] = self.test_df["target"].apply(lambda x: [x])
                self.ts_test_df["target"] = self.ts_test_df["target"].apply(
                    lambda x: [x]
                )

    def init_fltrust(self):
        if self.cfg.dataset == "cifar":
            self.fltrust_train_df = pd.read_csv(
                f"/{self.cfg.cifar.base_dir}/image_data/cifar/fltrust_train_map_file.csv"
            )
        else:
            self.fltrust_train_df = create_dataframe(
                self.cfg,
                "train_directories",
                dataset=self.cfg.federated_params.fltrust_dataset,
            )
        print(f"Initialized {self.cfg.federated_params.method} dataset\n", flush=True)

    def eval_fn(self, mode="eval"):
        self.model.eval()
        self.test_loss = 0
        fin_targets = []
        fin_outputs = []

        loader = self.valid_loader

        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=self.test_df,
        )

        if mode == "find_trust_score":
            loader = choose_correct_loader(
                self.cfg, self.ts_test_df, "valid", drop_last=False, shuffle=False
            )

            self.criterion = get_loss(
                loss_cfg=self.cfg.loss,
                device=self.device,
                df=self.ts_test_df,
            )

        with torch.no_grad():
            for bi, batch in enumerate(loader):
                index, (input, targets) = batch

                inp = self.get_model_input(input)
                targets = targets.to(self.device)

                outputs = self.model(inp)
                loss = self.get_loss_value(self.criterion, outputs, targets)

                self.test_loss += loss.detach().item()

                fin_targets.extend(targets.tolist())
                fin_outputs.extend(outputs.tolist())

        return fin_targets, fin_outputs

    def print_metrics(self):
        print(f"\nServer results:", flush=True)
        fin_targets, fin_outputs = self.eval_fn()
        if self.cfg.dataset == "cifar":
            calculate_cifar_metrics(fin_targets, fin_outputs)
        else:
            self.calculate_metrics(fin_targets, fin_outputs)

    def fltrust_train(self):

        if self.cfg.federated_params.opt_from_train:
            self.optimizer = instantiate(
                self.cfg.optimizer, params=self.model.parameters()
            )
        
        self.criterion = get_loss(
            loss_cfg=self.cfg.loss,
            device=self.device,
            df=self.fltrust_train_df,
        )

        states = []
        start = time.time()
        self.global_model_state = copy.deepcopy(self.model).state_dict()

        for _ in range(self.cfg.federated_params.round_epochs):
            old_state = copy.deepcopy(self.model)
            state = OrderedDict()
            self.model.train()

            for batch in self.train_loader:
                index, (input, targets) = batch

                inp = input[0].to(self.device)

                targets = targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inp)

                loss = self.get_loss_value(self.criterion, outputs, targets)

                loss.backward()

                self.optimizer.step()

            self.model.eval()
            for (key, weights1), (_, weights2) in zip(
                self.model.state_dict().items(), old_state.state_dict().items()
            ):
                state[key] = weights1 - weights2
            states.append(state)

        end = time.time()

        for key, weights in states[0].items():
            self.aggregated_state[key] = 0
        for state in states:
            for key, weights in state.items():
                self.aggregated_state[key] = self.aggregated_state[key] + weights

        self.model = get_model(self.cfg, len(self.cfg.ecg_record_params.leads))
        self.model.load_state_dict(self.global_model_state)

        print(
            f"Server finished training in {end - start} seconds, state is saved ({self.cfg.federated_params.method})",
            flush=True,
        )

    def fltrust_weight_update(self):
        print(flush=True)
        client_directions = []
        trust_scores = []
        server_direction = torch.cat(
            [x.flatten() for x in list(self.aggregated_state.values())]
        )

        for i in range(self.cfg.federated_params.amount_of_clients):
            client_directions.append(
                torch.cat([x.flatten() for x in list(self.states[i + 1].values())])
            )
            trust_scores.append(
                relu(
                    torch.dot(server_direction, client_directions[i])
                    / (torch.norm(server_direction) * torch.norm(client_directions[i]))
                )
            )
            print(f"Client {i + 1} trust score: {trust_scores[i]}", flush=True)

        for i in range(self.cfg.federated_params.amount_of_clients):
            modified_client_model_weights = OrderedDict()
            for key, weights in self.states[i + 1].items():
                modified_client_model_weights[key] = (
                    (1 / torch.stack(trust_scores, dim=0).sum(dim=0))
                    * trust_scores[i]
                    * (torch.norm(server_direction) / torch.norm(client_directions[i]))
                    * weights
                )
            self.states[i + 1] = modified_client_model_weights

    def fltrust_new_weight_update(self):
        print(flush=True)

        tmp_weights = self.global_model_state.copy()
        for key, weights in self.aggregated_state.items():
            tmp_weights[key] = tmp_weights[key] + weights
        self.model = get_model(self.cfg, len(self.cfg.ecg_record_params.leads))
        self.model.load_state_dict(tmp_weights)
        self.model.eval()

        targets, fin_outputs = self.eval_fn(mode="find_trust_score")
        server_result = self.activation(torch.as_tensor(fin_outputs))
        trust_scores = []

        for i in range(self.cfg.federated_params.amount_of_clients):

            tmp_weights = self.global_model_state.copy()
            for key, weights in self.states[i + 1].items():
                tmp_weights[key] = tmp_weights[key] + weights
            self.model = get_model(self.cfg, self.leads_num)
            self.model.load_state_dict(tmp_weights)
            self.model.eval()

            _, fin_outputs = self.eval_fn(mode="find_trust_score")
            client_result = self.activation(torch.as_tensor(fin_outputs))

            trust_score = []
            for j in range(len(server_result)):
                trust_score.append(
                    self.count_trust_score_for_new_fltrust(
                        server_result[j], client_result[j], targets[j]
                    )
                )

            trust_scores.append(
                (sum(trust_score) / self.cfg.federated_params.fltrust_new_sample_amount)
            )
            print(f"Client {i + 1} trust score: {trust_scores[i][0]}", flush=True)

        trust_scores = torch.Tensor(trust_scores).to(self.device)

        for i in range(self.cfg.federated_params.amount_of_clients):
            modified_client_model_weights = OrderedDict()
            for key, weights in self.states[i + 1].items():
                if sum(trust_scores):
                    modified_client_model_weights[key] = (
                        (1 / sum(trust_scores))
                        * trust_scores[i]
                        * weights
                        * self.cfg.federated_params.amount_of_clients
                    )
                else:
                    modified_client_model_weights[key] = 0
            self.states[i + 1] = modified_client_model_weights

        self.model = get_model(self.cfg, len(self.cfg.ecg_record_params.leads))
        self.model.load_state_dict(self.global_model_state)

    def count_trust_score_for_new_fltrust(self, server_result, client_result, target):
        if self.cfg.dataset == "cifar":
            client_result = torch.tensor([client_result[target]])
            server_result = torch.tensor([server_result[target]])
        return 2 * (1 - abs(client_result - server_result)) * (client_result + server_result) * (client_result + server_result) / 4

    def ts_momentum_weight_update(self):
        print(flush=True)

        old_state = self.model.state_dict().copy()

        _, _ = self.eval_fn(mode="find_trust_score")
        cur_loss = self.test_loss

        trust_scores = []
        uncutted_trust_scores = []

        for i in range(self.cfg.federated_params.amount_of_clients):

            tmp_weights = old_state.copy()
            for key, weights in self.states[i + 1].items():
                tmp_weights[key] = tmp_weights[key] + weights
            self.model = get_model(self.cfg, self.leads_num)
            self.model.load_state_dict(tmp_weights)
            self.model.eval()
            _, _ = self.eval_fn(mode="find_trust_score")
            client_loss = self.test_loss
            print(
                f"Client {i + 1} loss: {client_loss}, server loss: {cur_loss}",
                flush=True,
            )
            loss_diff = cur_loss - client_loss
            uncutted_trust_scores.append(loss_diff)
            trust_scores.append(max(loss_diff, 0))

        print(flush=True)

        for i in range(self.cfg.federated_params.amount_of_clients):
            if sum(trust_scores):
                momentum_trust_score = (
                    1 - self.momentum_beta
                ) * self.prev_trust_scores[i] + self.momentum_beta * (
                    trust_scores[i] / sum(trust_scores)
                )
            else:
                momentum_trust_score = (
                    1 - self.momentum_beta
                ) * self.prev_trust_scores[i]
            self.prev_trust_scores[i] = momentum_trust_score

        # Define proportion of trusted clients
        num_trusted_clients = int(
            self.cfg.federated_params.amount_of_clients
            * self.cfg.federated_params.ts_momentum_trusted_prop
        )
        trusted_ids = [
            idx
            for idx, _ in sorted(
                enumerate(uncutted_trust_scores), key=lambda x: x[1], reverse=True
            )[:num_trusted_clients]
        ]
        # Normalize trust_scores to simplex
        if self.cfg.federated_params.normalize_ts_momentum:
            self.prev_trust_scores = softmax(
                torch.tensor(self.prev_trust_scores, dtype=torch.float32), dim=0
            ).tolist()
        # Update modified_client_weights
        for i in range(self.cfg.federated_params.amount_of_clients):
            print(
                f"Client {i + 1} trust score: {self.prev_trust_scores[i]}", flush=True
            )
            modified_client_model_weights = OrderedDict()
            for key, weights in self.states[i + 1].items():
                if i in trusted_ids or trust_scores[i]:
                    modified_client_model_weights[key] = (
                        momentum_trust_score
                        * weights
                        * self.cfg.federated_params.amount_of_clients
                    )
                else:
                    modified_client_model_weights[key] = 0.0
            self.states[i + 1] = modified_client_model_weights

        self.model = get_model(self.cfg, len(self.cfg.ecg_record_params.leads))
        self.model.load_state_dict(old_state)

    def _get_opt_name(self):
        return type(self.optimizer).__name__

    def update_fedadam_vt(self):
        assert all(
            set(self.gradients[1].keys()) == set(grad.keys())
            for grad in self.gradients[1:]
        )
        # Averaging gradients
        average_gradient = {
            k: torch.zeros_like(v) for k, v in self.gradients[-1].items()
        }
        for i in range(len(self.gradients) - 1):
            for key, weights in self.gradients[i + 1].items():
                average_gradient[key] += weights * (
                    1 / self.cfg.federated_params.amount_of_clients
                )

        # Calculate diag(g x g) ?
        # Source code https://pytorch.org/docs/stable/_modules/torch/optim/adam.html#Adam contains `exp_avg_sqs` which is a ordered dict like model params.
        # In simple implementation from scratch v_t is also a dict with the same structure as gradient params
        # So, we do not calculate diag(g x g), we simply add g**2 to previous state

        # Initialization v_0 as average of v_0 from all clients
        all_zero_padded_v_t = all(torch.all(v == 0) for v in self.v_t.values())
        if all_zero_padded_v_t:
            for i in range(len(self.v_ts) - 1):
                for key, weights in self.v_ts[i + 1].items():
                    self.v_t[key] += weights * (
                        1 / self.cfg.federated_params.amount_of_clients
                    )
        # Update v_t
        for key, avg_grad in average_gradient.items():
            self.v_t[key] = (
                self.optimizer.vt_momentum * self.v_t[key]
                + (1 - self.optimizer.vt_momentum) * avg_grad**2
            )

    def recess_weight_update(self):
        print(flush=True)

        trust_scores = []
        for i in range(self.cfg.federated_params.amount_of_clients):
            trust_scores.append(
                self.init_trust_score
                - self.abnormality_alphas[i + 1] * self.baseline_decreased_score
            )

        softmax = torch.nn.Softmax()
        trust_scores = softmax(torch.tensor(trust_scores))

        for i in range(self.cfg.federated_params.amount_of_clients):
            print(f"Client {i + 1} trust score: {trust_scores[i]}", flush=True)
            modified_client_model_weights = OrderedDict()
            for key, weights in self.states[i + 1].items():
                modified_client_model_weights[key] = (
                    trust_scores[i]
                    * weights
                    * self.cfg.federated_params.amount_of_clients
                )
            self.states[i + 1] = modified_client_model_weights

    def perform_ipm_attack(self):
        grad_sum = OrderedDict()
        non_attacking_clients = [id for id in range(len(self.states)) if id not in self.attacking_clients and id != 0]

        for key in self.states[1]:
            grad_sum[key] = 0

        # Sum the gradients of non-attacking clients
        for id in non_attacking_clients:
            for key, weights in self.states[id].items():
                grad_sum[key] += weights

        # Update the states of attacking clients
        for id in self.attacking_clients:
            for key in self.states[id]:
                if not ("bn" in key or "running" in key):
                    self.states[id][key] = (-1) * self.ipm_eps * grad_sum[key] / len(non_attacking_clients)


def preprocess_dataset(cfg):
    print("Reading datasets...", flush=True)
    train_df = create_dataframe(cfg, "train_directories")
    valid_df = create_dataframe(cfg, "valid_directories")
    train_df = pd.concat([train_df, valid_df], ignore_index=True)
    list_of_clients = get_list_of_clients(
        train_df,
        cfg.federated_params.min_sample_number,
        cfg.task_params.pathology_names,
        cfg.federated_params.amount_of_clients,
        cfg.federated_params.client_sample,
    )
    list_of_clients = {
        client: index + 1 for index, client in enumerate(list_of_clients)
    }
    train_df = train_df[train_df["ID_CLINIC"].isin(list_of_clients.keys())]
    print("Preprocess successfull", flush=True)

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
            train_df = change_labels_of_clients(
                train_df,
                cfg.federated_params.attacking_method,
                list(list_of_clients.keys()),
                attacking_clients,
                cfg.federated_params.percent_of_changed_labels,
            )
            print(f"Succesfully flipped the labels", flush=True)

    train_df["ID_CLINIC"] = train_df["ID_CLINIC"].apply(lambda x: list_of_clients[x])

    return train_df


def choose_correct_loader(cfg, df, mode=None, drop_last=True, shuffle=True):
    if cfg.dataset == "cifar":
        loader = init_cifar_loader(df, mode, cfg.training_params.batch_size, shuffle=shuffle)
    else:
        loader = get_dataset_loader(df, cfg, None, cfg.task_params.classes, drop_last)
    return loader
