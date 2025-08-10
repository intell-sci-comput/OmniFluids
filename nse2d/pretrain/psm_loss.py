
import torch
import math
EPS = 1e-7


def PSM_NS_vorticity(w, v, t_interval=5.0, loss_mode='cn'):
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
    k_x = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(N, 1).repeat(1, N).reshape(1,N,N,1)
    k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=device),
                     torch.arange(start=-k_max, end=0, step=1, device=device)), 0).reshape(1, N).repeat(N, 1).reshape(1,N,N,1)
    # Negative Laplacian in Fourier space
    lap = k_x ** 2 + k_y ** 2
    lap[0, 0, 0, 0] = 1.0
    f_h = w_h / lap

    ux_h = 1j * k_y * f_h
    uy_h = -1j * k_x * f_h
    wx_h = 1j * k_x * w_h
    wy_h = 1j * k_y * w_h
    wlap_h = -lap * w_h

    ux = torch.fft.irfft2(ux_h[:, :, :k_max + 1], dim=[1, 2])
    uy = torch.fft.irfft2(uy_h[:, :, :k_max + 1], dim=[1, 2])
    wx = torch.fft.irfft2(wx_h[:, :, :k_max+1], dim=[1,2])
    wy = torch.fft.irfft2(wy_h[:, :, :k_max+1], dim=[1,2])
    wlap = torch.fft.irfft2(wlap_h[:, :, :k_max+1], dim=[1,2])
    dt = t_interval / (nt-1)

    if loss_mode=='cn':
        wt = (w[:, :, :, 1:] - w[:, :, :, :-1]) / ( dt)
        Du = (ux*wx + uy*wy - v.reshape(-1, 1, 1, 1)*wlap) #- forcing
        Du1 = wt + (Du[..., :-1] + Du[..., 1:]) * 0.5 
    if loss_mode=='mid':
        wt = (w[:, :, :, 2:] - w[:, :, :, :-2]) / (2 * dt)
        Du1 = wt + (ux*wx + uy*wy - v.reshape(-1, 1, 1, 1)*wlap)[...,1:-1] #- forcing
    return Du1


def PSM_loss(u, forcing, v, t_interval=0.50, loss_mode='cn'):
    batchsize = u.size(0)
    nx = u.size(1)
    ny = u.size(2)
    nt = u.size(3)
    u = u.reshape(batchsize, nx, ny, nt)
    Du = PSM_NS_vorticity(u, v, t_interval, loss_mode)
    if loss_mode == 'cn':
        forcing = forcing.reshape(-1, nx, ny, 1)
        f = forcing.repeat(1, 1, 1, nt-1)
    if loss_mode == 'mid':
       forcing = forcing.reshape(-1, nx, ny, 1)
       f = forcing.repeat(1, 1, 1, nt-2)
    return (torch.square(Du - f).mean() + EPS).sqrt()