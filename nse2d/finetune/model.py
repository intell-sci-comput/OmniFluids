import numpy as np
import torch
import copy
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

    
class FeedForward(nn.Module):
    def __init__(self, dim, factor, n_layers, layer_norm):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(n_layers):
            in_dim = dim if i == 0 else dim * factor
            out_dim = dim if i == n_layers - 1 else dim * factor
            self.layers.append(nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(inplace=True) if i < n_layers - 1 else nn.Identity(),
                nn.LayerNorm(out_dim) if layer_norm and i == n_layers -
                1 else nn.Identity(),
            ))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class SpectralConv2d_dy(nn.Module):
    def __init__(self, K, in_dim, out_dim, n_modes,
                 fourier_weight, factor, 
                 n_ff_layers, layer_norm):
        super().__init__()
        self.K = K
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_modes = n_modes
        self.fourier_weight = fourier_weight
        if not self.fourier_weight:
            self.fourier_weight = nn.ParameterList([])
            for _ in range(2):
                weight = torch.FloatTensor(K, in_dim, out_dim, n_modes, 2)
                param = nn.Parameter(weight)
                nn.init.xavier_normal_(param)
                self.fourier_weight.append(param)
        self.backcast_ff = FeedForward(out_dim, factor, n_ff_layers, layer_norm)


    def forward(self, x, att):
        x = self.forward_fourier(x, att)
        b = self.backcast_ff(x)
        return b

    def forward_fourier(self, x, att):
        x = rearrange(x, 'b m n i -> b i m n')
        B, I, M, N = x.shape
        weight = [torch.einsum("bk, kioxy->bioxy", att, self.fourier_weight[0]), 
                  torch.einsum("bk, kioxy->bioxy", att, self.fourier_weight[1])]
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        out_ft = x_fty.new_zeros(B, I, N, M // 2 + 1)
        out_ft[:, :, :, :self.n_modes] = torch.einsum(
                "bixy,bioy->boxy", x_fty[:, :, :, :self.n_modes],
                torch.view_as_complex(weight[0]))
        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        out_ft = x_ftx.new_zeros(B, I, N // 2 + 1, M)
        out_ft[:, :, :self.n_modes, :] = torch.einsum(
                "bixy,biox->boxy", x_ftx[:, :, :self.n_modes, :],
                torch.view_as_complex(weight[1]))

        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')
        x = xx + xy
        x = rearrange(x, 'b i m n -> b m n i')
        return x


class Student2D(nn.Module):
    def __init__(self, s=256, K=4, T=10, modes=16, width=60, output_dim=40,
                 n_layers=16, factor=4, n_ff_layers=2, layer_norm=True):
        super().__init__()
        self.modes = modes
        self.width = width
        self.s = s
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.backcast_ff = None
        self.fourier_weight = None
        self.K = K
        self.T = T

        grid = self.get_grid((1, s, s))
        self.register_buffer('grid', grid)
        self.in_proj = nn.Linear(5, self.width)

        self.f_nu = nn.ModuleList([])
        for _ in range(n_layers):
            self.f_nu.append(nn.Sequential(nn.Linear(1, 128),
                                     nn.GELU(), nn.Linear(128, 128),
                                     nn.GELU(), nn.Linear(128, self.K)))


        self.spectral_layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.spectral_layers.append(SpectralConv2d_dy(self.K, in_dim=width,
                                                       out_dim=width,
                                                       n_modes=modes,
                                                       fourier_weight=self.fourier_weight,
                                                       factor=factor,
                                                       n_ff_layers=n_ff_layers,
                                                       layer_norm=layer_norm))

        self.output_mlp_student = nn.Sequential(nn.Linear(self.width, 4 * self.width),
                            nn.GELU(),
                            nn.Linear(4 * self.width, 1)
                            ) 
        
    def forward(self, x, param, **kwargs):
        batch_size, size = x.shape[0], x.shape[1] 
        x_o = x
        x = torch.cat((x, self.grid.repeat(batch_size, 1, 1, 1)), dim=-1)
        x = torch.cat((x, param), dim=-1)
        x = self.in_proj(x)
        x = F.gelu(x)
        nu = param[:, 0, 0, 1:2]
        for i in range(self.n_layers):
            fc = self.f_nu[i]
            att = fc(nu)
            att = F.softmax(att/self.T, dim=-1)
            layer = self.spectral_layers[i]
            b = layer(x, att)
            x = x + b    
        x = F.gelu(b)
        x = self.output_mlp_student(x)
        x = x_o + x 
        return x

    def get_grid(self, shape, device='cpu'):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = 2 * torch.pi * torch.tensor(np.linspace(0, 1 - 1 / size_x, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = 2 * torch.pi * torch.tensor(np.linspace(0, 1 - 1 / size_y, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)