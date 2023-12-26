import torch

from algorithm.Sinkhorn_flow import Sinkhorn_flow

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
            t_iter = (t[i] - t_floor[i]) * torch.ones_like(t)
            t_iter = pad_t_like_x(t_iter, xt_flow[0])
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
    
    
        
        