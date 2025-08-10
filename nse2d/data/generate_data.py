import torch
import numpy as np
import random
import math
import os

from nse import navier_stokes_2d
from sampler import Init_generation, Force_generation
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     os.environ['PYTHONHASHSEED'] = str(seed) 
     torch.backends.cudnn.deterministic = True
     torch.backends.cudnn.benchmark = False
     torch.backends.cudnn.enabled = True


def generate_ns_data(cfg):
    seed = cfg.seed
    device = cfg.device
    s = cfg.s
    sub = cfg.sub

    dt = cfg.dt
    T = cfg.T
    record_steps = int(cfg.record_ratio * T)
    N = cfg.N
    name = cfg.name
    lognu_min, lognu_max = math.log10(cfg.re[0]), math.log10(cfg.re[1]) #[3, 6]
    amplitude_range = (-cfg.amplitude, cfg.amplitude)

    setup_seed(seed)
    save_path = f'./dataset'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    GRF = Init_generation(s, device=device)
    F_Sampler = Force_generation(s, max_frequency=cfg.max_frequency, amplitude_range=amplitude_range, device=device)

    bsize = 20
    if N < bsize:
        bsize = N
    c = 0
    u = torch.zeros(N, s//sub, s//sub, record_steps+1)
    param_record = torch.zeros(N, s//sub, s//sub, 2)

    for j in range(N//bsize):
        w0 = GRF(bsize)
        f = F_Sampler(bsize)
        visc = - lognu_min - (lognu_max - lognu_min) * torch.rand(bsize).to(device)
        sol, sol_t = navier_stokes_2d(w0, f, 10**visc, T, dt, record_steps)
        w0 = w0[:, ::sub, ::sub].reshape(-1,  s//sub, s//sub, 1)
        f = f[:, ::sub, ::sub].reshape(-1,  s//sub, s//sub, 1)
        sol = torch.concat([w0.to('cpu'), sol[:, ::sub, ::sub, :]], dim=3)
        u[c:(c+bsize),...] = sol
        param_record[c:(c+bsize),..., 0:1] = f
        param_record[c:(c+bsize),..., 1:2] = visc.reshape(-1, 1, 1, 1) * torch.ones_like(f)
        c += bsize
        print(j, c)
        print(u.max())
    torch.save(u, f'{save_path}/data_{name}')
    torch.save(param_record, f'{save_path}/f_{name}')
