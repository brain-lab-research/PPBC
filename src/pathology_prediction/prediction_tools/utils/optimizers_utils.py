import torch
from torch import optim


# Define custom optimizer 
class FedAdam(optim.Adam): 
    def __init__(self, params, lr=1e-3, weight_decay=0., betas=(0., 0.999), eps=1e-8, vt_momentum=0.99):
        super().__init__(params, lr=lr, betas=betas, eps=eps) 
        self.vt_momentum = vt_momentum
        self.weight_decay = weight_decay
        self.update_vt = False

    def update_v_t(self, v_t):
        """ Update second momentum from global Server

        Args:
            v_t (OrderedDict): second momentum with structure as `state['exp_avg_sq']` 
        """
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    self.state[p]['exp_avg_sq'] = v_t[group['name']]
        # Set flag as True to change `step()` behaviour
        self.update_vt = True

    def get_v_t(self):
        v_t = {}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    v_t[group['name']] = self.state[p]['exp_avg_sq']
        return v_t

    def step(self): 
        for group in self.param_groups: 
            for p in group['params']: 
                if p.grad is None: 
                    continue
                grad = p.grad.data 
                if grad.is_sparse: 
                    raise RuntimeError("Adam does not support sparse gradients") 

                state = self.state[p] 

                # State initialization 
                if len(state) == 0: 
                    state["step"] = 0
                    # Exponential moving average of gradient values 
                    state["exp_avg"] = torch.zeros_like(p.data) 
                    # Exponential moving average of squared gradient values 
                    state["exp_avg_sq"] = torch.zeros_like(p.data) 

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"] 
                beta1, beta2 = group["betas"]
                # First epoch as default Adam, next with constant `exp_avg_sq`
                if self.update_vt:
                    beta2 = 1.

                state["step"] += 1

                if self.weight_decay != 0: 
                    grad = grad.add(p.data, alpha=self.weight_decay) 

                # Decay the first and second moment running average coefficient 
                exp_avg.mul_(beta1).add_(1 - beta1, grad) 
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad) 

                denom = exp_avg_sq.sqrt().add_(group["eps"]) 

                if not self.update_vt:
                    bias_correction1 = torch.tensor(1 - beta1 ** state["step"])
                    bias_correction2 = torch.tensor(1 - beta2 ** state["step"]) 
                    step_size = group["lr"] * torch.sqrt(bias_correction2) / bias_correction1
                # With beta2 = 1 we can't use bias correction because we divide by zero assertion
                # If we don't use the correction, we apply constant `exp_avg_sq` as we want
                else:
                    step_size = group["lr"] 

                p.data.addcdiv_(-step_size, exp_avg, denom) 