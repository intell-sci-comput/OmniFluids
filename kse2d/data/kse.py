import torch
import math


def ks_2d_rk4(u0, T, param, dt=1e-3, record_steps=100):
    """
    2D Kuramoto-Sivashinsky equation solver using spectral method + RK4 time stepping.
    ∂u/∂t + 1/2 * (∇u)^2 + Δu + Δ^2 u = 0

    Parameters:
        u0: [batch, N, N] tensor, initial condition in physical space
        T: total simulation time
        dt: time step
        record_steps: number of saved steps (default 100)
    Returns:
        sol: [N, N, record_steps] tensor of u in physical space
        times: [record_steps] tensor of time values
    """
    device = u0.device
    param = param.to(device)
    batch = u0.shape[0]
    N = u0.shape[-1]
    steps = math.ceil(T / dt)
    save_every = steps // record_steps

    # Wavenumbers
    k_max = N/2
    kx = 1/4 * torch.fft.fftfreq(N, d=1.0 / N).to(device)
    ky = 1/4 * torch.fft.fftfreq(N, d=1.0 / N).to(device)
    kx, ky = torch.meshgrid(kx, ky, indexing="ij")

    # Spectral operators
    lap = -(kx ** 2 + ky ** 2)
    lap2 = lap ** 2
    L = lap + lap2  # Linear operator
    L = - L[None, :] * param[:, 0].reshape(batch, 1, 1)

    # Initialize
    u_h = torch.fft.fft2(u0)
    sol = torch.zeros(batch, N, N, record_steps+1, device='cpu')
    sol[..., 0] = u0
    t = 0.0
    c = 1

    def nonlinear(u_phys):
        dealias = torch.unsqueeze(torch.logical_and(torch.abs(ky) <= (2.0/3.0)*k_max, torch.abs(kx) <= (2.0/3.0)*k_max).float(), 0)[None, :]
        ux = torch.fft.ifft2(1j * kx * torch.fft.fft2(u_phys)).real
        uy = torch.fft.ifft2(1j * ky * torch.fft.fft2(u_phys)).real
        return -0.5 * param[:, 1].reshape(batch, 1, 1) * torch.fft.fft2(ux ** 2 + uy ** 2) * dealias

    for i in range(steps):
        u_phys = torch.fft.ifft2(u_h).real
        N1 = nonlinear(u_phys)
        k1 = dt * (L * u_h + N1)

        u_phys = torch.fft.ifft2(u_h + 0.5 * k1).real
        N2 = nonlinear(u_phys)
        k2 = dt * (L * (u_h + 0.5 * k1) + N2)

        u_phys = torch.fft.ifft2(u_h + 0.5 * k2).real
        N3 = nonlinear(u_phys)
        k3 = dt * (L * (u_h + 0.5 * k2) + N3)

        u_phys = torch.fft.ifft2(u_h + k3).real
        N4 = nonlinear(u_phys)
        k4 = dt * (L * (u_h + k3) + N4)

        u_h = u_h + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        t += dt

        if (i + 1) % save_every == 0:
            sol[:, :, :, c] = torch.fft.ifft2(u_h).real.cpu()
            print(f"Step {i + 1}/{steps}, t={t:.4f}, max(u)={sol[:,:,:,c].max():.4f}")
            c += 1

    return sol