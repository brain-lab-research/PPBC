from collections import OrderedDict

from ..base.fedavg import FedAvg
from .ts_momentum_server import TSMomentumServer
from utils.data_utils import read_dataframe_from_cfg, get_stratified_subsample


class TSMomentum(FedAvg):
    def __init__(self, trust_sample_amount, momentum_beta, trusted_prop):
        super().__init__()
        self.trust_sample_amount = trust_sample_amount
        self.momentum_beta = momentum_beta
        self.trusted_prop = trusted_prop

    def _init_server(self, cfg):
        trust_df = read_dataframe_from_cfg(cfg, "train_directories", "trust_df")
        _, trust_df = get_stratified_subsample(
            df=trust_df,
            num_samples=self.trust_sample_amount,
            random_state=cfg.random_state,
        )
        self.num_clients = cfg.federated_params.amount_of_clients
        self.prev_trust_scores = [1 / self.num_clients] * self.num_clients
        self.server = TSMomentumServer(cfg, trust_df)

    def _count_trust_score(self, server_loss, client_losses):
        # Calculate current trust scores
        uncutted_trust_scores = [
            server_loss - client_loss for client_loss in client_losses
        ]
        trust_scores = [
            max(uncutted_trust_score, 0)
            for uncutted_trust_score in uncutted_trust_scores
        ]
        # Update trust scores with momentum
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
        # Define proportion of trusted clients
        num_trusted_clients = int(self.num_clients * self.trusted_prop)
        trusted_ids = [
            idx
            for idx, _ in sorted(
                enumerate(uncutted_trust_scores), key=lambda x: x[1], reverse=True
            )[:num_trusted_clients]
        ]
        return trusted_ids, trust_scores

    def _modify_gradients(self, trusted_ids, trust_scores):
        # Modify each client's gradient based on there trust score
        print("\nTrust Score Calculation")
        for i in range(self.num_clients):
            if i in trusted_ids:
                current_ts = 1.0
            elif trust_scores[i]:
                current_ts = self.prev_trust_scores[i] * self.num_clients
            else:
                current_ts = 0.0
            print(f"Client {i} trust score: {current_ts / self.num_clients}")
            modified_client_model_weights = {
                k: v * current_ts for k, v in self.server.client_gradients[i].items()
            }
            self.server.client_gradients[i] = modified_client_model_weights

    def aggregate(self):
        server_loss, client_losses = self.server.get_trust_losses()
        trusted_ids, trust_scores = self._count_trust_score(server_loss, client_losses)
        self._modify_gradients(trusted_ids, trust_scores)
        aggregated_weights = super().aggregate()
        return aggregated_weights
