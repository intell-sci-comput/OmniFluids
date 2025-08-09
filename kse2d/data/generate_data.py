import torch
import numpy as np
import random
import math
import os
import json
import sys
from kse import *
from sampler import Init_generation
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




def generate_ks_data(cfg):
    device = cfg.device
    s = cfg.s
    sub = cfg.sub
    dt = cfg.dt
    T = cfg.T
    record_steps = int(cfg.record_ratio * T)
    mode = cfg.mode

    if mode == 'train':
        setup_seed(0)
        cfg.N = 2
    if mode == 'test':
        setup_seed(1)
        cfg.N = 10
    if mode == 'val':
        setup_seed(2)
        cfg.N = 2
    N = cfg.N
    param1, param2 = cfg.param1, cfg.param2       
    data_save_path = f'./dataset'
    if not os.path.exists(data_save_path):
        os.makedirs(data_save_path)
    
    GRF = Init_generation(s, device=device)
    bsize = min(100, N)
    c = 0
    u = torch.zeros(N, s//sub, s//sub, record_steps+1)
    param_record = torch.zeros(N, 2)

    for j in range(N//bsize):
        # Sample random feilds
        w0 = GRF(bsize)
        param = torch.ones(bsize, 2, device=device)
        param[:, 0] = (param1[1] - param1[0]) * torch.rand(bsize, device=device) + param1[0]
        param[:, 1] = (param2[1] - param2[0]) * torch.rand(bsize, device=device) + param2[0]
        sol = ks_2d_rk4(w0, T, param, dt=dt, record_steps=record_steps)
        sol = sol[:, ::sub, ::sub, :].to('cpu')
        u[c:(c+bsize),...] = sol.to('cpu')
        param_record[c:(c+bsize),...] = param.to('cpu')
        c += bsize
        print(j, c)
        print(u.max())
    torch.save(u, f'{data_save_path}/{cfg.name}')
    torch.save(param_record, f'{data_save_path}/param_{cfg.name}')