import torch
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Init_generation(object):

    def __init__(self, size, L1=2 * math.pi, L2=2 * math.pi, alpha=4, tau=8.0, sigma=None, mean=None, boundary="periodic", device=None, dtype=torch.float64):

        s1, s2 = size, size
        self.s1 = s1
        self.s2 = s2

        self.mean = mean

        self.device = device
        self.dtype = dtype

        if sigma is None:
            self.sigma = 0.5 * tau**(0.5*(2*alpha - 2.0))
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
