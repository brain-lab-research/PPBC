from ..ts_momentum.ts_momentum import TSMomentum
from collections import OrderedDict

import numpy as np
import torch
from utils.attack_utils import (
    map_attack_clients,
    set_attack_rounds,
    set_client_map_round,
    load_attack_configs,
)

import time

class EFMomentum(TSMomentum):
    def __init__(self, trust_sample_amount, momentum_beta, trusted_prop, theta, gamma, **politic_args):
        super().__init__(trust_sample_amount, momentum_beta, trusted_prop)
        self.theta = theta
        self.gamma = gamma
        # self.method = politic_args.get('method', 'top-k')
        self.k = politic_args.get('k', 3)
        
    def _init_federated(self, cfg, df):
        super()._init_federated(cfg, df)
        
        self.amount_of_clients = self.cfg.federated_params.amount_of_clients
        self.current_errors_from_clients = {f'client {i}' : OrderedDict() for i in range(self.amount_of_clients)}
        self.final_errors = {f'client {i}' : OrderedDict() for i in range(self.amount_of_clients)}
        self.iterations = 1
        
    def compress(self):
        amount = self.amount_of_clients - self.k

        idx_of_k_clients = np.argsort(torch.tensor(self.prev_trust_scores))[:amount]
        self.compress_politic = torch.zeros_like(self.current_politic)
        
        for i in range(amount):
            self.compress_politic[idx_of_k_clients[i]] = self.current_politic[idx_of_k_clients[i]]
        print(self.prev_trust_scores, self.compress_politic, 'trust scores of clients and politic for round')
        
    def get_init_point(self):
        aggregated_weights = self.server.global_model.state_dict()
        for rank in range(self.amount_of_clients):
            client_errors = self.final_errors[f'client {rank}']

            for key, _ in aggregated_weights.items():
                aggregated_weights[key] = _ + client_errors[key]
                
        self.server.global_model.load_state_dict(aggregated_weights) 
    
    
    def get_errors_on_iter(self, itn):
        aggregated_weights = self.server.global_model.state_dict()
        
        for rank in range(self.amount_of_clients):
                client_grad = self.server.client_gradients[rank]
                current_client_error = self.current_errors_from_clients[f'client {rank}']
                final_client_error = self.final_errors[f'client {rank}']
                client_politic = self.compress_politic[rank].to(self.server.device)
                
                for key, grads in client_grad.items():
                    # print(current_client_error[key].device, client_politic.device, grads.device, flush=True)
                    self.current_errors_from_clients[f'client {rank}'][key] = current_client_error[key] + \
                        (1 - self.theta) * (1/self.amount_of_clients - client_politic) * \
                        grads.to(self.server.device)

                    aggregated_weights[key] = aggregated_weights[key] + \
                        self.gamma * (1 - self.theta) * grads.to(self.server.device) * client_politic + \
                            self.gamma * self.theta * final_client_error[key]
                print(f'client {rank} has finished')
                            
                if itn == self.iterations-1:
                    self.final_errors[f'client {rank}'] = self.current_errors_from_clients[f'client {rank}']
            
        print('final errors saved!')
        return aggregated_weights
    
    def init_errors(self):
        if self.cur_round != 0:
            self.compress()
            for rank in range(self.amount_of_clients):
                for key, _ in self.server.global_model.state_dict().items():
                    self.current_errors_from_clients[f'client {rank}'][key] = torch.zeros_like(_) .to(self.server.device) 
            return
        else:
            self.current_politic = torch.ones(self.amount_of_clients).to(self.server.device) / self.amount_of_clients
            self.compress()
            for rank in range(self.amount_of_clients):
                for key, _ in self.server.global_model.state_dict().items():
                    self.current_errors_from_clients[f'client {rank}'][key] = torch.zeros_like(_) .to(self.server.device) 
                    self.final_errors[f'client {rank}'][key] = torch.zeros_like(_).to(self.server.device)
            print('creating the errors is done')
                    
            return
        
    def process_clients(self):
        
        self.init_errors()
        self.get_init_point()
        
        for itn in range(self.iterations):
            self.client_map_round = set_client_map_round(
                self.client_map, self.attack_rounds, self.attack_scheme, self.cur_round
            )
            
            print(f'start the {itn} iteration')
            super().train_round()
            
            server_loss, client_losses = self.server.get_trust_losses()
            trusted_ids, trust_scores = self._count_trust_score(server_loss, client_losses)
            
            aggregated_weights = self.get_errors_on_iter(itn)
             
            self.server.global_model.load_state_dict(aggregated_weights)
    
            print('processing is done')
            
    def begin_train(self):
        self.create_clients()
        self.clients_loader = self.manager.batches
        
        for round in range(self.rounds):
            print(f"\nRound number: {round} of {self.rounds}")
            begin_round_time = time.time()
            self.cur_round = round

            metrics = self.server.test_global_model()

            print("\nTraining started\n")

            self.client_map_round = set_client_map_round(
                self.client_map, self.attack_rounds, self.attack_scheme, round
            )
            
            self.process_clients()
            # super().train_round()
            
            self.server.save_best_model(round)
            for i in range(self.amount_of_clients):
                c = 0
                for key, w in self.final_errors[f'client {i}'].items():
                    if np.allclose(w.cpu(), torch.zeros_like(w).cpu()):
                        c += 1
                if c == len(self.final_errors[f'client {i}'].items()):
                    print(f'all errors for {i} client are equals to zeros')
            
            print(f"Round time: {time.time() - begin_round_time}", flush=True)
            
        self.stop_train()
        
        