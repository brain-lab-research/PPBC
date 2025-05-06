from ..ts_momentum.ts_momentum import TSMomentumServer

from collections import OrderedDict

import numpy as np
import random
import torch

from utils.attack_utils import (
    map_attack_clients,
    set_attack_rounds,
    set_client_map_round,
    load_attack_configs,
)

import time

from ..base.fedavg import FedAvg
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

from utils.data_utils import read_dataframe_from_cfg, get_stratified_subsample


class EF25(FedAvg):
    def __init__(self, theta, gamma, distribution=None, **method_args):
        '''
        need_ef:bool - do we need calculate errors or just do fedavg-like strategy
        method:str - random, bant, gradient_norm, loss
        distribution:list - size of dataset for each client
        k:int - amount of clients that we use 
        '''
        super().__init__()
    
        self.theta = theta
        self.gamma = gamma
        self.distribution = distribution
        
        self.epoch_method = method_args.get('epoch_method', 'random')
        self.k = method_args.get('k', 3)
        self.iterations = method_args.get('iterations', 1)
        self.need_errors = method_args.get('need_errors', 'True')
        
        self.trust_sample_amount = method_args.get('trust_sample_amount', 50)
        self.momentum_beta = method_args.get('momentum_beta', 0.1)
        
    def _init_server(self, cfg):
        trust_df = read_dataframe_from_cfg(cfg, "train_directories", "trust_df")
        _, trust_df = get_stratified_subsample(
            df=trust_df,
            num_samples=self.trust_sample_amount,
            random_state=cfg.random_state,
        )
        
        self.server = TSMomentumServer(cfg, trust_df)
        
        self.num_clients = cfg.federated_params.amount_of_clients
        self.prev_trust_scores = [1 / self.num_clients] * self.num_clients
        
        assert self.k <= self.num_clients, 'k cannot be bigger than amount of clients'
        
    def _init_federated(self, cfg, df):
        super()._init_federated(cfg, df)

        self.current_errors_from_clients = {f'client {i}' : OrderedDict() for i in range(self.num_clients)}
        self.final_errors = {f'client {i}' : OrderedDict() for i in range(self.num_clients)}
        
    def get_scores_from_gradients(self):
        for rank in range(self.num_clients):
            cur_grads = self.server.client_gradients[rank]
            cur_flat_grad = torch.cat([grad.flatten() for grad in cur_grads.values()])
            
            self.prev_trust_scores[rank] =  torch.norm(cur_flat_grad)
            
    def get_scores_from_bant(self, server_loss, client_losses):
        uncutted_trust_scores = [
            server_loss - client_loss for client_loss in client_losses
        ]
        trust_scores = [
            max(uncutted_trust_score, 0)
            for uncutted_trust_score in uncutted_trust_scores
        ]

        for i in range(self.num_clients):
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
            
    def get_scores_from_losses(self):
        client_results = self.server.server_metrics
        self.prev_trust_scores = [metrics[1] for metrics in client_results]
    
    def _epoch_count_trust_score(self):
        if 'bant' in self.epoch_method:
            server_loss, client_losses = self.server.get_trust_losses()
            self.get_scores_from_bant(server_loss, client_losses)
        elif 'gradient_norm' in self.epoch_method:
            self.get_scores_from_gradients()
        elif 'loss' in self.epoch_method:
            self.get_scores_from_losses()
        else:
            print(f'{self.epoch_method} method does not requires trust scores')
                   
    def random_compressor(self):
        clients = np.arange(self.num_clients)
        random.shuffle(clients)
        
        self.compress_politic = torch.zeros_like(self.current_politic)
        for rank in range(self.k):
            self.compress_politic[clients[rank]] = self.current_politic[clients[rank]]
            
        print(clients, self.compress_politic, 'perm of clients and politic for round')
    
    def trust_score_compressor(self):
        idx_of_k_clients = np.argsort(self.prev_trust_scores)[:self.k]
        
        self.compress_politic = torch.zeros_like(self.current_politic)
        for rank in range(self.k):
            self.compress_politic[idx_of_k_clients[rank]] = self.current_politic[idx_of_k_clients[rank]]
        
        print(self.prev_trust_scores, self.compress_politic, f'trust scores via {self.epoch_method} of clients and politic for round')
            
    
    def epoch_compressor(self):
        if 'random' in self.epoch_method:
            self.random_compressor()
        else:
            self.trust_score_compressor()
              
        
    def get_init_point(self):
        aggregated_weights = self.server.global_model.state_dict()
        for rank in range(self.num_clients):
            client_errors = self.final_errors[f'client {rank}']

            for key, _ in aggregated_weights.items():
                aggregated_weights[key] = _ + client_errors[key]
                
        self.server.global_model.load_state_dict(aggregated_weights) 
        
    def get_data_size(self):
        current_data_size = torch.sum(self.compress_politic * torch.tensor(self.distribution).to(self.server.device) * self.num_clients) 
        return current_data_size
    
    
    def get_errors_on_iter(self, itn):
        aggregated_weights = self.server.global_model.state_dict()
        
        data_size = self.get_data_size()
        print(f'now we use {data_size} objects from dataset')
        
        for rank in range(self.num_clients):
                client_grad = self.server.client_gradients[rank]
                current_client_error = self.current_errors_from_clients[f'client {rank}']
                
                final_client_error = self.final_errors[f'client {rank}']
                client_politic = self.compress_politic[rank].to(self.server.device)
                
                for key, grads in client_grad.items():
                    # print(current_client_error[key].device, client_politic.device, grads.device, flush=True)
                    self.current_errors_from_clients[f'client {rank}'][key] = current_client_error[key] + \
                        (1 - self.theta) * (1/self.num_clients - client_politic) * \
                        grads.to(self.server.device) * int(self.need_errors)

                    aggregated_weights[key] = aggregated_weights[key] + \
                        self.gamma * (1 - self.theta) * grads.to(self.server.device) * client_politic * int(self.need_errors)  + \
                            self.gamma * self.theta * final_client_error[key] * int(self.need_errors) +\
                        self.gamma * (1 - self.theta) * grads.to(self.server.device) * client_politic * int(not self.need_errors) * (self.distribution[rank] / data_size)
                # print(f'client {rank} has finished')
                if self.need_errors:            
                    if itn == self.iterations-1:
                        self.final_errors[f'client {rank}'] = self.current_errors_from_clients[f'client {rank}']
            
        print('final errors saved!')
        return aggregated_weights
    
    def init_errors(self):
        if self.cur_round != 0:
            self.epoch_compressor()
            for rank in range(self.num_clients):
                for key, _ in self.server.global_model.state_dict().items():
                    self.current_errors_from_clients[f'client {rank}'][key] = torch.zeros_like(_) .to(self.server.device) 
            return
        else:
            self.current_politic = torch.ones(self.num_clients).to(self.server.device) / self.num_clients
            self.epoch_compressor()
            for rank in range(self.num_clients):
                for key, _ in self.server.global_model.state_dict().items():
                    self.current_errors_from_clients[f'client {rank}'][key] = torch.zeros_like(_) .to(self.server.device) 
                    self.final_errors[f'client {rank}'][key] = torch.zeros_like(_).to(self.server.device)
            print('creating the errors is done')
                    
            return
        
    def process_clients(self):
        self.init_errors()
        
        if self.need_errors:
            self.get_init_point()
        
        for itn in range(self.iterations):
            print(f'start the {itn} iteration')
            super().train_round()
            
            aggregated_weights = self.get_errors_on_iter(itn)
             
            self.server.global_model.load_state_dict(aggregated_weights)
    
            print('processing is done')
            
        self._epoch_count_trust_score()
        
    def check_final_errors(self):
        for i in range(self.num_clients):
            c = 0
            for key, w in self.final_errors[f'client {i}'].items():
                if np.allclose(w.cpu(), torch.zeros_like(w).cpu()):
                    c += 1
            if c == len(self.final_errors[f'client {i}'].items()):
                print(f'all errors for {i} client are equals to zeros')
        
        
    def begin_train(self):
        self.create_clients()
        self.clients_loader = self.manager.batches
        
        for round in range(self.rounds):
            print(f"\nRound number: {round} of {self.rounds}")
            begin_round_time = time.time()
            self.cur_round = round

            _ = self.server.test_global_model()

            print("\nTraining started\n")

            self.client_map_round = set_client_map_round(
                self.client_map, self.attack_rounds, self.attack_scheme, round
            )
            
            self.process_clients()
            # super().train_round()
            
            self.server.save_best_model(round)
            
            self.check_final_errors()
            
            print(f"Round time: {time.time() - begin_round_time}", flush=True)
            
            
        self.stop_train()
        
        
            
        
    