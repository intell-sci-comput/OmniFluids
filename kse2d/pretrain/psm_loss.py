
import torch
import math
EPS = 1e-7


def PSM_KS(w, param, t_interval=5.0, loss_mode='cn'):
    batchsize = w.size(0)
    nx = w.size(1)
    ny = w.size(2)
    nt = w.size(3)
    device = w.device
    w = w.reshape(batchsize, nx, ny, nt)

    w_h = torch.fft.fft2(w, dim=[1, 2])
    # Wavenumbers in y-direction
    k_max = nx//2
    N = nx
    k_x = 1/4 * torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1, N).reshape(1,N,N,1)
    k_y = 1/4 * torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N, 1).reshape(1,N,N,1)
    # Negative Laplacian in Fourier space
    lap = - (k_x ** 2 + k_y ** 2)
    L = (lap + lap ** 2) * param[:, 0].reshape(-1, 1 , 1, 1) * w_h

    wx = torch.fft.ifft2(1j * k_x * w_h, dim=[1, 2]).real
    wy = torch.fft.ifft2(1j * k_y * w_h, dim=[1, 2]).real
    N = 0.5 * param[:, 1].reshape(-1, 1 , 1, 1) * torch.fft.fft2(wx ** 2 + wy ** 2, dim=[1, 2]) 
    dt = t_interval / (nt-1)
    if loss_mode=='cn':
        wt = (w[:, :, :, 1:] - w[:, :, :, :-1]) / dt
        Du = torch.fft.ifft2(L + N, dim=[1, 2]).real
        Du1 = wt + (Du[..., :-1] + Du[..., 1:]) * 0.5 
    if loss_mode=='mid':
        wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)
        Du1 = wt + torch.fft.ifft2(L + N, dim=[1, 2]).real[...,1:-1] #- forcing
    return Du1

def PSM_loss(u, param, t_interval=0.50, loss_mode='cn'):
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)
    u = u.reshape(batchsize, nx, ny, nt)
    Du = PSM_KS(u, param, t_interval, loss_mode)
    return (torch.square(Du).mean() + EPS).sqrt()