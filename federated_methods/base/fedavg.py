import time
import torch.multiprocessing as mp

from .server import Server
from .client import Client, multiprocess_client
from utils.manager_utils import Manager
from utils.attack_utils import (
    map_attack_clients,
    set_attack_rounds,
    set_client_map_round,
    load_attack_configs,
)


class FedAvg:
    def __init__(self):
        self.server = None
        self.client = None
        self.rounds = 0
        self.terminated_event = None
        self.client_map_round = None

    def _init_federated(self, cfg, df):
        self.cfg = cfg
        self.df = df
        # Initialize the server, and client's base
        self.attack_setup(cfg)
        self._init_server(cfg)
        self._init_client_cls()
        self._init_manager()

    def _init_server(self, cfg):
        self.server = Server(cfg)

    def _init_client_cls(self):
        self.client_cls = Client
        self.client_args = [self.cfg, self.df]
        self.client_kwargs = {
            "client_cls": self.client_cls,
            "pipe": None,
            "rank": None,
            "attack_type": None,
        }

    def _init_manager(self):
        self.manager = Manager(self.cfg, self.server)

    def attack_setup(self, cfg):
        self.rounds = cfg.federated_params.communication_rounds
        self.client_map = map_attack_clients(
            cfg.federated_params.clients_attack_types,
            cfg.federated_params.prop_attack_clients,
            cfg.federated_params.amount_of_clients,
        )
        self.attack_scheme = cfg.federated_params.attack_scheme
        self.attack_rounds = set_attack_rounds(
            cfg.federated_params.prop_attack_rounds, self.rounds, self.attack_scheme
        )
        self.attack_configs = load_attack_configs(
            cfg, cfg.federated_params.clients_attack_types
        )

    def aggregate(self):
        aggregated_weights = self.server.global_model.state_dict()
        for i in range(len(self.server.client_gradients)):
            for key, weights in self.server.client_gradients[i].items():
                aggregated_weights[key] = aggregated_weights[key] + weights.to(
                    self.server.device
                ) * (1 / len(self.server.client_gradients))
        return aggregated_weights

    def create_clients(self):
        self.processes = []

        # Init pipe for every client
        self.pipes = [mp.Pipe() for _ in range(self.manager.batches.batch_size)]
        self.server.pipes = [pipe[0] for pipe in self.pipes]  # Init input (server) pipe

        for rank in range(self.manager.batches.batch_size):
            # Every process starts by calling the same function with the given arguments
            self.client_kwargs["pipe"] = self.pipes[rank][1]  # Send current pipe
            self.client_kwargs["rank"] = rank
            self.client_kwargs["attack_type"] = self.client_map[rank]
            p = mp.Process(
                target=multiprocess_client,
                args=self.client_args,
                kwargs=self.client_kwargs,
            )
            p.start()
            self.processes.append(p)

    def get_communication_content(self, rank):
        # In fedavg we need to send model after aggregate and
        # attack type for every client
        return {
            "update_model": {
                k: v.cpu() for k, v in self.server.global_model.state_dict().items()
            },
            "attack_type": (
                self.client_map_round[rank],
                self.attack_configs[self.client_map_round[rank]],
            ),
        }

    def parse_communication_content(self, client_result):
        # In fedavg we recive result_dict from every client
        self.server.set_client_result(client_result)
        print(f"Client {client_result['rank']} finished in {client_result['time']}")
        if self.cfg.federated_params.print_client_metrics:
            # client_result['client_metrics'] = (loss, metrics)
            client_loss, client_metrics = (
                client_result["client_metrics"][0],
                client_result["client_metrics"][1],
            )
            print(client_metrics)
            print(f"Validation loss: {client_loss}\n")

    def train_round(self):
        for batch_idx, clients_batch in enumerate(self.clients_loader):
            print(f"Current batch of clients is {clients_batch}", flush=True)

            # Send content to clients to start local learning
            for pipe_num, rank in enumerate(clients_batch):
                content = self.get_communication_content(rank)
                self.server.send_content_to_client(pipe_num, content)

            # Waiting end of local learning and recieve content from clients
            for pipe_num, rank in enumerate(clients_batch):
                content = self.server.rcv_content_from_client(pipe_num)
                self.parse_communication_content(content)

            # Manager reinit clients with new ranks
            self.manager.step(batch_idx)

    def stop_train(self):
        # Send to all clients message to shutdown
        for rank in range(self.manager.batches.batch_size):
            self.server.shutdown_client(rank)

        for rank, p in enumerate(self.processes):
            p.join()

    def begin_train(self):
        self.create_clients()
        self.clients_loader = self.manager.batches

        for round in range(self.rounds):
            print(f"\nRound number: {round}")
            begin_round_time = time.time()
            self.cur_round = round

            self.server.test_global_model()

            print("\nTraining started\n")

            self.client_map_round = set_client_map_round(
                self.client_map, self.attack_rounds, self.attack_scheme, round
            )

            self.train_round()

            self.server.save_best_model(round)

            aggregated_weights = self.aggregate()
            self.server.global_model.load_state_dict(aggregated_weights)

            print(f"Round time: {time.time() - begin_round_time}", flush=True)

        print("Shutdown clients, federated learning end", flush=True)
        self.stop_train()
