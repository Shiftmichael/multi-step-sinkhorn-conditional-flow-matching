import os
import math
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss
from algorithm.optimal_transport import OTPlanSampler


class SD(object):
    def __init__(self, blur, scaling, x0, init_mass) ->  None:
        self.particles = x0
        self.velocity = None
        self.mass = init_mass
        self.potential_op = SamplesLoss(
            loss = 'sinkhorn', p = 2, blur = blur, potentials = True, 
            debias = False, backend = "online", scaling = scaling
        )


    def one_step_update(self, step_size = None, x1 = None, tgt_mass = None, noise_scale = 0, **kw):
        self.particles.requires_grad = True
        first_var_ab, _ = self.potential_op(
            self.mass, self.particles, tgt_mass, x1
        )
        first_var_aa, _ = self.potential_op(
            self.mass, self.particles, self.mass, self.particles 
        )
        first_var_ab_grad = torch.autograd.grad(
            torch.sum(first_var_ab), self.particles
        )[0]
        first_var_aa_grad = torch.autograd.grad(
            torch.sum(first_var_aa), self.particles
        )[0]
        with torch.no_grad():
            vector_field = first_var_ab_grad - first_var_aa_grad
            self.velocity = vector_field
            noise = torch.randn_like(vector_field)
            self.particles = self.particles - step_size * vector_field + math.sqrt(2* step_size * noise_scale) * noise 

        self.particles.requires_grad = False
        torch.cuda.empty_cache()

    @torch.no_grad()
    def SD_clear_all(self):
        self.particles = None
    
    @torch.no_grad()
    def get_state(self):
        return self.particles
    
    @torch.no_grad()
    def get_v(self):
        return self.velocity

   
class Sinkhorn_flow(object):
    def __init__(self, x0, x1, **kw) -> None:
        super().__init__()
        self.x1 = x1
        self.tgt_mass = torch.ones(x1.shape[0], device = x1.device) / x1.shape[0]
        self.x0 = x0
        self.init_mass = torch.ones(x1.shape[0], device = x1.device) / x1.shape[0]
        self.record_support = []
        self.record_velocity = []
        self.ot_sampler = OTPlanSampler(method="exact")

    def forward(self, blur, scaling, steps, stepsize):
        algorithm = SD(blur = blur, scaling = scaling, x0 = self.x0, init_mass = self.init_mass)
        self.record_support.append(self.x0)
        support = None
        for step in range(steps - 1):
            lr = stepsize
            algorithm.one_step_update(
                step_size = lr,
                x1 = self.x1,
                tgt_mass = self.tgt_mass
            )
            support = algorithm.get_state()
            v = algorithm.get_v()
            self.record_support.append(support)  #[time, batch_size, 3*32*32]
            self.record_velocity.append(v)
        _, x1_new = self.ot_sampler.sample_plan(support, self.x1)
        self.record_support.append(x1_new)
        self.record_velocity.append(x1_new - support)
        algorithm.SD_clear_all()
        del algorithm

    @torch.no_grad()
    def get_state(self):
        return torch.stack(self.record_support).detach()
    
    @torch.no_grad()
    def get_v(self):
        return torch.stack(self.record_velocity).detach()
    
    @torch.no_grad()
    def sinkhorn_clear_all(self):
        self.record_support = []
        self.record_velocity = []
        
        
class Sinkhorn_gradient_decent(object):
    def __init__(self, x0, x1, **kw) -> None:
        super().__init__()
        self.x1 = x1
        self.tgt_mass = torch.ones(x1.shape[0], device = x1.device) / x1.shape[0]
        self.x0 = x0
        self.init_mass = torch.ones(x1.shape[0], device = x1.device) / x1.shape[0]
        self.record_support = []
        self.record_velocity = []

    def forward(self, blur, scaling, steps, stepsize):
        algorithm = SD(blur = blur, scaling = scaling, x0 = self.x0, init_mass = self.init_mass)
        self.record_support.append(self.x0)
        support = None
        for step in range(steps):
            lr = stepsize
            algorithm.one_step_update(
                step_size = lr,
                x1 = self.x1,
                tgt_mass = self.tgt_mass
            )
            support = algorithm.get_state()
            v = algorithm.get_v()
            if step != steps-1:
                self.record_support.append(support)  #[time, batch_size, 3*32*32]
            self.record_velocity.append(v)
        algorithm.SD_clear_all()
        del algorithm

    @torch.no_grad()
    def get_state(self):
        return torch.cat(self.record_support).detach()
    
    @torch.no_grad()
    def get_v(self):
        return torch.cat(self.record_velocity).detach()
    
    @torch.no_grad()
    def sinkhorn_clear_all(self):
        self.record_support = []
        self.record_velocity = []