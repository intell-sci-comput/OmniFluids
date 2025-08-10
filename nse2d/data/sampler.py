import torch
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class Init_generation(object):

    def __init__(self, size, L1=2 * math.pi, L2=2 * math.pi, alpha=2.5, tau=7.0, sigma=None, mean=None, boundary="periodic", device=None, dtype=torch.float64):

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

class Force_generation(object):

    def __init__(self, size, max_frequency=4, amplitude_range=(-0.1, 0.1), device=None):
        self.dim = 2
        self.device = device
        x = 2 * torch.pi * torch.linspace(0, 1 - 1/size, size, device=device)
        y = 2 * torch.pi * torch.linspace(0, 1 - 1/size, size, device=device)
        self.xx, self.yy = torch.meshgrid(x, y, indexing='xy')

        self.u = torch.arange(1, max_frequency+1, device=device)  # 频率 u 的范围
        self.v = torch.arange(1, max_frequency+1, device=device)  # 频率 v 的范围
        self.frequencies = torch.cartesian_prod(self.u, self.v).float()

        self.amplitude_range = amplitude_range
        self.max_frequency = max_frequency

    def __call__(self, N):
        amplitudes_real = torch.distributions.Uniform(*self.amplitude_range).sample((N, self.frequencies.shape[0],))
        amplitudes_real = amplitudes_real.to(self.device)
        amplitudes_imaginary = torch.distributions.Uniform(*self.amplitude_range).sample((N, self.frequencies.shape[0],))
        amplitudes_imaginary = amplitudes_imaginary.to(self.device)

        force_real = torch.sum(
            amplitudes_real[:, :, None, None] * torch.cos((self.frequencies[None, :, 0, None, None] * self.xx 
                                                                      + self.frequencies[None, :, 1, None, None] * self.yy)), dim=1)

        force_imaginary = torch.sum(
            amplitudes_imaginary[:, :, None, None] * torch.sin((self.frequencies[None, :, 0, None, None] * self.xx
                                                                     + self.frequencies[None, :, 1, None, None] * self.yy)), dim=1)

        force = force_real + force_imaginary
        force = force / self.max_frequency
        return force

