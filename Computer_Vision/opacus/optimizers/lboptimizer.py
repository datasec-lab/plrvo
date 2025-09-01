from opacus.optimizers.optimizer import _check_processed_flag, _mark_as_processed, DPOptimizer
from torch.distributions.laplace import Laplace
from torch.distributions.gamma import Gamma
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch import stack, zeros, einsum
from opacus.optimizers.utils import params
from torch import nn
from torch.optim import Optimizer
from scipy.stats import truncnorm, expon
from typing import Callable, Optional
import torch

class LapBiasOptimizer(DPOptimizer):
    """
    Implementation of PLRV first noise mechanism.
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        *,
        noise_multiplier: float,
        max_grad_norm: float,
        expected_batch_size: Optional[int],
        loss_reduction: str = "mean",
        generator=None,
        secure_mode: bool = False,
    ):
        super().__init__(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            generator=generator,
            secure_mode=secure_mode,
        )
        
    def make_noise(self, args):
        self.b = args['b']
        self.bias = args['bias']
        
    
    def add_noise(self):
        count = 0
        for p in self.params:
            _check_processed_flag(p.summed_grad)
            
            laplace = self.get_laplace(p.size())
            noise = laplace.sample(p.size()).to(p.summed_grad.device)
            p.grad = (p.summed_grad + noise)
            get_bias_back = torch.where(p.grad >= 0, 1, -1)
            #p.grad = p.grad - (self.bias*get_bias_back)  
            _mark_as_processed(p.summed_grad)
            
    def get_linear_combination(self, shape):
        return self.b*(self.max_grad_norm/self.bias)
        
    def get_laplace(self, shape):
        return Laplace(loc=0, scale=self.get_linear_combination(shape))
        
    def clip_and_accumulate(self):
        """
        Performs gradient clipping.
        Stores clipped and aggregated gradients into `p.summed_grad```
        """

        if len(self.grad_samples[0]) == 0:
            # Empty batch
            per_sample_clip_factor = torch.zeros(
                (0,), device=self.grad_samples[0].device
            )
        else:
            per_param_norms = [
                g.reshape(len(g), -1).norm(2, dim=-1) for g in self.grad_samples
            ]
            per_sample_norms = torch.stack(per_param_norms, dim=1).norm(2, dim=1)
            per_sample_clip_factor = (
                self.max_grad_norm / (per_sample_norms + 1e-6)
            ).clamp(max=1.0)

        for p in self.params:
            _check_processed_flag(p.grad_sample)
            grad_sample = self._get_flat_grad_sample(p)
            grad = torch.einsum("i,i...", per_sample_clip_factor, grad_sample)
            #grad = torch.where(grad >= 0, grad+self.bias, grad-self.bias)
            if p.summed_grad is not None:
                p.summed_grad += grad
            else:
                p.summed_grad = grad

            _mark_as_processed(p.grad_sample)
