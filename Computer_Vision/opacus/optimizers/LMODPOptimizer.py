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

class PLRVDPOptimizer(DPOptimizer):
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
        self.gamma = None
        self.uniform = None
        self.normal = None
        self.args = args
        #print(self.args['k'])
        #print(self.args['theta'])
        if 'gamma' in args.keys():
            if args['gamma']:
                self.k = self.args['k']
                self.theta = self.args['theta']
                #self.bias = self.args['bias']
                self.gamma = Gamma(
                    concentration = self.k, rate = 1/self.theta
                )
        if 'uniform' in args.keys():
            if args['uniform']:
                self.a = self.args['a']
                self.b = self.args['b']
                self.uniform = Uniform(
                    low = self.a, high = self.b
                )
        if 'truncnorm' in args.keys():
            if args['truncnorm']:
                self.mu = self.args['mu']
                self.sigma = self.args['sigma']
                self.l = self.args['l']
                self.u = self.args['u']
                a_transformed = (self.l - self.mu) / self.sigma 
                b_transformed = (self.u - self.mu) / self.sigma
                self.normal= truncnorm(
                    a_transformed, b_transformed, loc=self.mu, scale=self.sigma
                )
        
        self.clip = self.args['max_grad_norm']
        #self.lam = self.args['lam']
        
        #self.expon = expon(loc=0, scale = 1/self.lam)
        #self.laplace = self.get_laplace()
        
    
    def add_noise(self):
        count = 0
        for p in self.params:
            _check_processed_flag(p.summed_grad)
            
            self.gradients = torch.abs(p)
            laplace = self.get_laplace(p.size())
            #noise = laplace.sample(p.summed_grad.shape).to(p.summed_grad.device)
            noise = laplace.sample().to(p.summed_grad.device)
            #noise = 0
            #print(noise.size() == p.summed_grad.size())
            #noise = [max(i, self.max_grad_norm) for j in i for i in noise]
            #print(p.summed_grad[0])
            p.grad = (p.summed_grad + (noise))
            #get_bias_back = torch.where(p.grad >= 0, 1, -1)
            #print(get_bias_back[0])
            #p.grad = p.grad - (self.bias*get_bias_back)
            #p.grad = p.grad/max((1, p.grad.norm(2)/(self.max_grad_norm*3)))
            #if(max((1, p.grad.norm(1)/self.max_grad_norm)) > 1):
            #    count += 1
                
            _mark_as_processed(p.summed_grad)
        #print(count/len(self.params))
            
    def get_linear_combination(self, shape):
        den = 0
        if self.gamma is not None:
            self.gamma = Gamma(concentration = self.k, rate = 1/(self.theta))
            den += self.gamma.sample(shape)
            #print(den.mean())
        #if self.uniform is not None:
        #    den += self.uniform.sample(shape)
       # if self.normal is not None:
       #     den += self.normal.rvs(size=1)[0]  
        #exp = self.expon.rvs(size=1)[0]
        return 1/(den)

        
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
