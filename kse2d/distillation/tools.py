import matplotlib.pyplot as plt
import torch
import numpy as np
import math
import random
import os
from functools import reduce
import operator
from thop import profile
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

def parse_filename(cfg, filename):
    parts = filename.split('-')
    for i, part in enumerate(parts):
        if part == 'seed':
            cfg.seed = int(parts[i + 1])
        elif part == 'K':
            cfg.K = int(parts[i + 1])
        elif part == 'T':
            cfg.T = float(parts[i + 1])
        elif part == 'modes':
            cfg.modes = int(parts[i + 1])
        elif part == 'width':
            cfg.width = int(parts[i + 1])
        elif part == 'n_layers':
            cfg.n_layers = int(parts[i + 1])
        elif part == 'input_dim':
            cfg.input_dim = int(parts[i + 1])
        elif part == 'share_weight':
            cfg.share_weight = parts[i + 1] == 'True'
        elif part == 'size':
            cfg.size = int(parts[i + 1])


def param_flops(net):
    device = next(net.parameters()).device
    dummy_input = torch.randn(1, 128, 128, net.output_dim, device=device)
    dummy_input2 = torch.randn(1, 128, 128, 2, device=device)

    params = 0
    for p in list(net.parameters()):
        params += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    print(' params: %.3f M' % (params / 1000000.0))
    return params

class Init_generation(object):

    def __init__(self, size, L1=2 * math.pi, L2=2 * math.pi, alpha=4, tau=8.0, sigma=None, mean=None, boundary="periodic", device=None, dtype=torch.float64):

        s1, s2 = size, size
        self.s1 = s1
        self.s2 = s2

        self.mean = mean

        self.device = device
        self.dtype = dtype

        if sigma is None:
            self.sigma = tau**(0.5*(2*alpha - 2.0))
        else:
            self.sigma = sigma

        const1 = (4*(math.pi**2))/(L1**2)
        const2 = (4*(math.pi**2))/(L2**2)

        freq_list1 = torch.cat((torch.arange(start=0, end=s1//2, step=1),\
                                torch.arange(start=-s1//2, end=0, step=1)), 0)
        k1 = freq_list1.view(-1,1).repeat(1, s2//2 + 1).type(dtype).to(device)

        freq_list2 = torch.arange(start=0, end=s2//2 + 1, step=1)

        k2 = freq_list2.view(1,-1).repeat(s1, 1).type(dtype).to(device)

        self.sqrt_eig = s1*s2*self.sigma*((const1*k1**2 + const2*k2**2 + tau**2)**(-alpha/2.0))
        self.sqrt_eig[0,0] = 0.0

    def __call__(self, N, xi=None):
        if xi is None:
            xi  = torch.randn(N, self.s1, self.s2//2 + 1, 2, dtype=self.dtype, device=self.device)
        
        xi[...,0] = self.sqrt_eig*xi [...,0]
        xi[...,1] = self.sqrt_eig*xi [...,1]
        
        u = torch.fft.irfft2(torch.view_as_complex(xi), s=(self.s1, self.s2))

        if self.mean is not None:
            u += self.mean
        
        return u 
