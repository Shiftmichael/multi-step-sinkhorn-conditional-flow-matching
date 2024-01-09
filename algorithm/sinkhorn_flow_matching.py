import random
import torch
from geomloss import SamplesLoss

from algorithm.Sinkhorn_flow import Sinkhorn_flow, Sinkhorn_gradient_decent

def pad_t_like_x(t, x):
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

class multistepSinkhornflowFlowMatching:
    
    def __init__(self, blur, scaling, steps, stepsize) -> None:
        self.blur = blur
        self.scaling = scaling
        self.steps = steps
        self.stepsize = stepsize
         
    def compute_sf_flow(self, x0, x1):
        sf_flow = Sinkhorn_flow(x0, x1)
        sf_flow.forward(self.blur, self.scaling, self.steps, self.stepsize)
        xt_flow = sf_flow.get_state()
        return xt_flow
        
    def compute_xt_ut(self, xt_flow, t, t_floor, t_ceil):
        xt = []
        ut = []
        for i in range(t.shape[0]):
            index = torch.randint(0, xt_flow[0].shape[0], (1,))
            # t_iter = (t[i] - t_floor[i]) * torch.ones_like(t)
            # t_iter = pad_t_like_x(t_iter, xt_flow[0])
            t_iter = t[i] - t_floor[i]
            xt_i = t_iter * xt_flow[t_ceil[i]] + (1 - t_iter) * xt_flow[t_floor[i]]
            ut_i = xt_flow[t_ceil[i]] - xt_flow[t_floor[i]]
            xt.append(xt_i[index])
            ut.append(ut_i[index])
        return torch.cat(xt), torch.cat(ut)
            
    def sample_location_and_conditional_flow(self, x0, x1, t = None):
        xt_flow = self.compute_sf_flow(x0, x1)
        device = x0.device
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"
        
        t = t * self.steps # [0, 1] -> [0, steps]
        t_floor = torch.floor(t).to('cpu', torch.int) # lower rounding
        t_ceil = torch.ceil(t).to('cpu', torch.int) # upper rounding
        
        xt, ut = self.compute_xt_ut(xt_flow, t, t_floor.to(device), t_ceil.to(device))

        return t, xt, ut
    
    def multi_sample_location_and_conditional_flow(self, x0, x1, t = None, multi_samples = 1):
        xt_flow = self.compute_sf_flow(x0, x1)
        device = x0.device
        t_multi = []
        xt_multi = []
        ut_multi = []
        for _ in range(multi_samples):
            t = torch.rand(x0.shape[0]).type_as(x0)
            t = t * self.steps # [0, 1] -> [0, steps]
            t_floor = torch.floor(t).to('cpu', torch.int) # lower rounding
            t_ceil = torch.ceil(t).to('cpu', torch.int) # upper rounding
            
            xt, ut = self.compute_xt_ut(xt_flow, t, t_floor.to(device), t_ceil.to(device))
            
            t_multi.append(t)
            xt_multi.append(xt)
            ut_multi.append(ut)

        return torch.stack(t_multi), torch.stack(xt_multi), torch.stack(ut_multi)
    
    
    
class SinkhornvelocityFlowmatcher:
    
    def __init__(self, blur, scaling) -> None:
        self.velocity_op = SamplesLoss(
            loss = 'sinkhorn', p = 2, blur = blur, potentials = True, 
            debias = False, backend = "online", scaling = scaling
        )
    
    def compute_ut(self, xt, x1):
        tgt_mass = torch.ones(x1.shape[0], device = x1.device) / x1.shape[0]
        init_mass = torch.ones(xt.shape[0], device = xt.device) / xt.shape[0]
        # assert tgt_mass == init_mass , "mass has to be the same"
        batch_size = xt.shape[0]
        xt = xt.view(batch_size, -1)
        x1 = x1.view(batch_size, -1)
        xt.requires_grad = True
        first_var_ab, _ = self.velocity_op(
            init_mass, xt, tgt_mass, x1
        )
        first_var_aa, _ = self.velocity_op(
            init_mass, xt, init_mass, xt 
        )
        first_var_ab_grad = torch.autograd.grad(
            torch.sum(first_var_ab), xt
        )[0]
        first_var_aa_grad = torch.autograd.grad(
            torch.sum(first_var_aa), xt
        )[0]
        with torch.no_grad():
            vector_field = first_var_ab_grad - first_var_aa_grad
        xt.requires_grad = False
        
        return vector_field
    
    def sample_xt(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        return t * x1 + (1 - t) * x0
    
    def sample_location_and_conditional_flow(self, x0, x1, t=None):
        if t is None:
            t = torch.rand(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"
        
        xt = self.sample_xt(x0, x1, t)
        ut = self.compute_ut(xt, x1).reshape(xt.shape)
        try:
            ut = (ut / torch.norm(ut, p=2)) * torch.norm(x1 - x0, p=2)
        except:
            ut = ut
        return t, xt, ut
    
    
class SinkhornFlowMatcher:
    
    def __init__(self, blur, scaling, steps, stepsize):
        self.blur = blur
        self.scaling = scaling
        self.steps = steps
        self.stepsize = stepsize
 
    def sample_location_and_conditional_flow(self, x0, x1, n):
        sf_flow = Sinkhorn_flow(x0, x1)
        sf_flow.forward(self.blur, self.scaling, self.steps, self.stepsize)
        xt_flow = sf_flow.get_state()
        vt_flow = sf_flow.get_v()
        v_return = []
        x_return = []
        t_return = []
        for i in range(n):
            xt = []
            vt = []
            t_batch = torch.randint(0, self.steps, (x0.shape[0],)).type_as(x0)
            t_return.append(t_batch)
            for i in range(t_batch.shape[0]):
                flag = int(t_batch[i].item())
                xt.append(xt_flow[flag][i])
                vt.append(vt_flow[flag][i])
            v_return.append(torch.stack(vt))
            x_return.append(torch.stack(xt))
        sf_flow.sinkhorn_clear_all()
        return torch.stack(t_return), torch.stack(x_return), torch.stack(v_return)
    


class SinkhorndecentwithFlowMatcher:
    
    def __init__(self, blur, scaling, steps, stepsize):
        self.blur = blur
        self.scaling = scaling
        self.steps = steps
        self.stepsize = stepsize
 
    def sample_location_and_conditional_flow(self, x0, x1, n):
        sf_flow = Sinkhorn_flow(x0, x1)
        sf_flow.forward(self.blur, self.scaling, self.steps, self.stepsize)
        xt_flow = sf_flow.get_state()
        vt_flow = sf_flow.get_v()
        v_return_SD = []
        x_return_SD = []
        t_return_SD = []
        for i in range(n):
            xt = []
            vt = []
            t_batch = torch.randint(0, self.steps-1, (x0.shape[0],)).type_as(x0)
            # t_batch = torch.randint(0, self.steps, (x0.shape[0],)).type_as(x0)
            t_return_SD.append(t_batch)
            for i in range(t_batch.shape[0]):
                flag = int(t_batch[i].item())
                xt.append(xt_flow[flag][i])
                vt.append(vt_flow[flag][i])
            v_return_SD.append(torch.stack(vt))
            x_return_SD.append(torch.stack(xt))
        v_return_FM = []
        x_return_FM = []
        t_return_FM = []
        for i in range(n):
            t = torch.rand(x0.shape[0]).type_as(x0) 
            t_real = t + self.stepsize * (self.steps - 1)
            t_return_FM.append(t_real)
            t = pad_t_like_x(t, x0)
            xt_batch =  t * xt_flow[self.steps] + (1 - t) * xt_flow[self.steps-1]
            vt_batch = vt_flow[self.steps-1]
            x_return_FM.append(xt_batch)
            v_return_FM.append(vt_batch)
        sf_flow.sinkhorn_clear_all()
        return torch.stack(t_return_SD), torch.stack(x_return_SD), torch.stack(v_return_SD),torch.stack(t_return_FM), torch.stack(x_return_FM), torch.stack(v_return_FM)
    
    
class traditionSinkhornFlowMatcher:
    
    def __init__(self, blur, scaling, steps, stepsize):
        self.blur = blur
        self.scaling = scaling
        self.steps = steps
        self.stepsize = stepsize
 
    def sample_location_and_conditional_flow(self, x0, x1, n):
        sf_flow = Sinkhorn_gradient_decent(x0, x1)
        sf_flow.forward(self.blur, self.scaling, self.steps, self.stepsize)
        xt_flow = sf_flow.get_state()
        vt_flow = sf_flow.get_v()
        batch_size = x0.shape[0]
        index = torch.randperm(xt_flow.size(0)).to(x0.device)
        v = vt_flow[index]
        x = xt_flow[index]
        v_return = []
        x_return = []
        t_return = []
        for i in range(n):
            t_batch = index[i * batch_size : (i + 1) * batch_size] // batch_size
            t_return.append(t_batch)
            xt = x[i * batch_size : (i + 1) * batch_size, :]
            vt = v[i * batch_size : (i + 1) * batch_size, :]
            v_return.append(vt)
            x_return.append(xt)
        sf_flow.sinkhorn_clear_all()
        return torch.stack(t_return), torch.stack(x_return), torch.stack(v_return)