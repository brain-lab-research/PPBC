import os
import time
import math
import hydra
import torch
import random
import warnings
import numpy as np
import pandas as pd
import torch.distributed as dist
from omegaconf import DictConfig
from torch.multiprocessing import Process

from utils.model_utils import get_model
from utils.data_utils import create_dataframe, change_labels_of_clients
from utils.multiprocess_utils import Client, Server, preprocess_dataset
from utils.cifar_utils import read_cifar_dataset

warnings.filterwarnings("ignore")

os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(random.randint(30000, 60000))


def multiprocess(rank: int, cfg: DictConfig, train_df: pd.DataFrame, backend="gloo"):

    dist.init_process_group(
        backend, rank=rank, world_size=cfg.federated_params.amount_of_clients + 1
    )
    client, server = 0, 0
    if rank:
        client = 1
    else:
        server = 1

    device = "cpu"
    if cfg.training_params.device == "cuda":
        device = "{}:{}".format(
            cfg.training_params.device, cfg.training_params.device_ids[0]
        )

    if (
        cfg.federated_params.attack
        and cfg.federated_params.attacking_method == "random_rounds_random_clients"
    ):
        if server:
            attacking_rounds = random.sample(
                list(range(cfg.federated_params.communication_rounds)),
                int(
                    cfg.federated_params.communication_rounds
                    * cfg.federated_params.amount_of_attack_rounds
                ),
            )
            attacking_rounds.sort()
            print(f"Attacking rounds: {attacking_rounds}", flush=True)
            attacking_rounds = torch.Tensor(attacking_rounds)
        else:
            attacking_rounds = torch.Tensor(
                [
                    0.0
                    for _ in range(
                        int(
                            cfg.federated_params.communication_rounds
                            * cfg.federated_params.amount_of_attack_rounds
                        )
                    )
                ]
            )

        dist.barrier()
        dist.broadcast(attacking_rounds, src=0)

    if server:
        base_class = Server(cfg)

        if cfg.dataset == "cifar":
            base_class.test_df = pd.read_csv(
                f"/{cfg.cifar.base_dir}/image_data/cifar/test_map_file.csv"
            )
        else:
            base_class.test_df = create_dataframe(
                cfg, "ptbxl_test_data", dataset="test_ptbxl"
            )
            pathology_name_to_save = (
                list(cfg.task_params.merge_map.keys())[0]
                if cfg.task_params.merge_map
                else cfg.task_params.pathology_names[0]
            )

        if "FLTrust" in cfg.federated_params.method:
            base_class.init_fltrust()

        base_class.init_loaders()

    elif client:
        base_class = Client(cfg)

        base_class.rank = rank

        if cfg.dataset == "cifar":
            base_class.train_df = train_df[train_df["client"] == rank]
        else:
            base_class.train_df = train_df[train_df["ID_CLINIC"] == rank]
        if cfg.federated_params.attack:
            if cfg.federated_params.attacking_method == "random_rounds_random_clients":
                base_class.attacked_train_df = change_labels_of_clients(
                    base_class.train_df.copy(),
                    cfg.federated_params.attacking_method,
                    p=cfg.federated_params.percent_of_changed_labels,
                    dataset_type="cifar",
                )
            elif (
                cfg.federated_params.attacking_method == "sign_flip"
                or cfg.federated_params.attacking_method == "random_grad"
                or cfg.federated_params.attacking_method == "delayed_grad"
            ):
                np.random.seed(cfg.federated_params.random_state)
                attacking_clients = np.random.choice(
                    range(1, cfg.federated_params.amount_of_clients + 1),
                    size=int(
                        cfg.federated_params.amount_of_clients
                        * cfg.federated_params.amount_of_attackers
                    ),
                    replace=False,
                )
                if rank == 1:
                    print(
                        f"Attack this round by these clients: {attacking_clients}\n",
                        flush=True,
                    )
                if rank in attacking_clients:
                    base_class.attacking = True
        base_class.init_loaders()

    base_class.device = device
    dist.barrier()

    for round in range(cfg.federated_params.communication_rounds):
        if server:
            print(
                f"===== Round number: {round} =====\n\nClients started training...",
                flush=True,
            )
            if (
                "FLTrust" in cfg.federated_params.method
                or cfg.federated_params.method == "TS_momentum"
            ):
                print(
                    f"Server started training... ({cfg.federated_params.method})\n",
                    flush=True,
                )

        start = time.time()

        if cfg.federated_params.attack and (
            (
                cfg.federated_params.attacking_method == "random_rounds_random_clients"
                and round in attacking_rounds
            )
            or cfg.federated_params.attacking_method == "IPM"
        ):
            if server:
                attacking_clients = np.random.choice(
                    range(1, cfg.federated_params.amount_of_clients + 1),
                    size=int(
                        cfg.federated_params.amount_of_clients
                        * cfg.federated_params.amount_of_attackers
                    ),
                    replace=False,
                )
                print(
                    f"Attack this round by these clients: {attacking_clients}\n",
                    flush=True,
                )
                attacking_clients = torch.Tensor(attacking_clients)
            else:
                attacking_clients = torch.Tensor(
                    [
                        0.0
                        for _ in range(
                            int(
                                cfg.federated_params.amount_of_clients
                                * cfg.federated_params.amount_of_attackers
                            )
                        )
                    ]
                )

            dist.barrier()
            dist.broadcast(attacking_clients, src=0)
            base_class.attacking = rank in attacking_clients

            if cfg.federated_params.attacking_method == "IPM":
                base_class.attacking_clients = attacking_clients.int()

        for key, _ in base_class.model.state_dict().items():
            dist.broadcast(base_class.model.state_dict()[key], src=0)

        # Broadcast Adam's second momentum to all clients
        if base_class.opt_name == "FedAdam":
            for key, _ in base_class.v_t.items():
                dist.broadcast(base_class.v_t[key], src=0)

        if server and "FLTrust" in cfg.federated_params.method:
            base_class.fltrust_train()
        elif client:
            base_class.current_com_round = round
            base_class.train()

        dist.barrier()
        dist.gather_object(
            base_class.aggregated_state, base_class.states if rank == 0 else None, dst=0
        )

        if (
            server
            and cfg.federated_params.attack
            and cfg.federated_params.attacking_method == "IPM"
        ):
            base_class.perform_ipm_attack()

        # Gather gradients from clients to server
        if base_class.opt_name == "FedAdam":
            dist.gather_object(
                base_class.aggregated_gradients,
                base_class.gradients if rank == 0 else None,
                dst=0,
            )
        # Gather second momentumn from clients on first round as v_0 initialization
        if base_class.opt_name == "FedAdam" and round == 0:
            dist.gather_object(
                base_class.v_t, base_class.v_ts if rank == 0 else None, dst=0
            )

        if client and cfg.federated_params.method == "RECESS":
            base_class.recess_detection()
        if cfg.federated_params.method == "RECESS":
            dist.barrier()
            dist.gather_object(
                base_class.abnormality_alpha,
                base_class.abnormality_alphas if rank == 0 else None,
                dst=0,
            )

        if server:
            aggregated_weights = base_class.model.state_dict()

            if cfg.federated_params.method == "FLTrust":
                base_class.fltrust_weight_update()
            elif cfg.federated_params.method == "FLTrust_new":
                base_class.fltrust_new_weight_update()
            elif cfg.federated_params.method == "TS_momentum":
                base_class.ts_momentum_weight_update()
            elif cfg.federated_params.method == "RECESS":
                base_class.recess_weight_update()

            for i in range(cfg.federated_params.amount_of_clients):
                for key, weights in base_class.states[i + 1].items():
                    aggregated_weights[key] = aggregated_weights[key] + weights * (
                        1 / cfg.federated_params.amount_of_clients
                    )

            # Averaging gradients from clients and update second momentum g^2
            if base_class.opt_name == "FedAdam":
                base_class.update_fedadam_vt()

            base_class.model.load_state_dict(aggregated_weights)
            base_class.model.eval()
            print(f"\nSuccessfully aggregated client's weights", flush=True)

        dist.barrier()
        end = time.time()

        for i in list(range(1, cfg.federated_params.amount_of_clients + 1)) + [0]:
            dist.barrier()
            if rank == i:
                if cfg.federated_params.print_client_metrics or server:
                    base_class.print_metrics()

        if server:
            if round == 0:
                base_class.best_round = (0, base_class.test_loss)
            elif base_class.test_loss < base_class.best_round[1] or math.isnan(
                base_class.best_round[1]
            ):
                base_class.best_round = (round, base_class.test_loss)

            if round == base_class.best_round[0]:
                if cfg.dataset == "cifar":
                    checkpoint_path = (
                        f"{cfg.single_run_dir}/cifar_{cfg.federated_params.method}.pt"
                    )
                else:
                    checkpoint_path = f"{cfg.single_run_dir}/{len(cfg.ecg_record_params.leads)}_leads_{cfg.federated_params.method}_{pathology_name_to_save}.pt"
                torch.save(aggregated_weights, checkpoint_path)

            print(
                f"\nRound {round} finished in {time.time() - start} seconds\n",
                flush=True,
            )
            print(
                f"Current test_loss: {base_class.test_loss}\nBest test_loss: {base_class.best_round[1]}\nBest round: {base_class.best_round[0]}\n",
                flush=True,
            )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def run(cfg: DictConfig):

    print("Checkpoint path: ", cfg.single_run_dir, end="\n\n")
    if cfg.dataset == "cifar":
        train_df = read_cifar_dataset(cfg)
    else:
        train_df = preprocess_dataset(cfg)
    processes = []
    for rank in range(cfg.federated_params.amount_of_clients + 1):
        p = Process(target=multiprocess, args=[rank, cfg, train_df])
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    run()
