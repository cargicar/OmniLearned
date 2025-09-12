"""Script taken from https://arxiv.org/pdf/2509.07700""" 

import numpy as np
import torch
from scipy import integrate
from tqdm import trange

from src.diffusion.dpm_solver import (
    DpmSolver,
    NoiseScheduleKarras,
    sample_dpmpp_2m_sde,
    sample_dpmpp_3m_sde,
    sample_dpmpp_sde,
)
from src.utils import append_dims


def to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    return (x - denoised) / append_dims(sigma, x.ndim)


@torch.no_grad()
def sample_euler(
    denoiser,
    x,
    sigmas,
    callback=None,
    progress=False,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """Implements Algorithm 2 (Euler steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=not progress):
        gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        dt = sigmas[i + 1] - sigma_hat
        x = x + d * dt
    return x


@torch.no_grad()
def sample_heun(
    denoiser,
    x,
    sigmas,
    callback=None,
    progress=False,
    s_churn=0.0,
    s_tmin=0.0,
    s_tmax=float("inf"),
    s_noise=1.0,
):
    """Implements Algorithm 2 (Heun steps) from Karras et al. (2022)."""
    s_in = x.new_ones([x.shape[0]])
    for i in trange(len(sigmas) - 1, disable=not progress):
        gamma = min(s_churn / (len(sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigmas[i] <= s_tmax else 0.0
        eps = torch.randn_like(x) * s_noise
        sigma_hat = sigmas[i] * (gamma + 1)
        if gamma > 0:
            x = x + eps * (sigma_hat**2 - sigmas[i] ** 2) ** 0.5
        denoised = denoiser(x, sigma_hat * s_in)
        d = to_d(x, sigma_hat, denoised)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigma_hat, "denoised": denoised})
        dt = sigmas[i + 1] - sigma_hat
        if sigmas[i + 1] == 0:
            # Euler method
            x = x + d * dt
        else:
            # Heun's method
            x_2 = x + d * dt
            denoised_2 = denoiser(x_2, sigmas[i + 1] * s_in)
            d_2 = to_d(x_2, sigmas[i + 1], denoised_2)
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x


def linear_multistep_coeff(t, r, n, j):
    """
    Approximates the integral of the j-th Lagrange polynomial of order r over the interval [t[n], t[n + 1]].
    We do not use explicit formula for j-th Adams-Bashforth coefficients, as we take steps of different sizes.
    See also: https://en.wikipedia.org/wiki/Linear_multistep_method#Adams-Bashforth_methods
    """
    if r - 1 > n:
        raise ValueError(f"Order {r} too high for step {n}")

    def l(tau):
        prod = 1.0
        for i in range(r):
            if i != j:
                prod *= (tau - t[n - i]) / (t[n - j] - t[n - i])
        return prod

    return integrate.quad(l, t[n], t[n + 1], epsrel=1e-4)[0]


@torch.no_grad()
def sample_linear_multistep(denoiser, x, sigmas, callback=None, progress=False, order=4):
    """Implements Linear Multistep a.k.a Adams-Bashforth method."""
    s_in = x.new_ones([x.shape[0]])
    sigmas_cpu = sigmas.detach().cpu().numpy()
    ds = []
    for i in trange(len(sigmas) - 1, disable=not progress):
        sigma = sigmas[i]
        denoised = denoiser(x, sigma * s_in)
        d = to_d(x, sigma, denoised)
        ds.append(d)
        if len(ds) > order:
            ds.pop(0)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigma, "sigma_hat": sigma, "denoised": denoised})
        cur_order = min(i + 1, order)
        coeffs = [linear_multistep_coeff(sigmas_cpu, cur_order, i, j) for j in range(cur_order)]
        x = x + sum(coeff * d for coeff, d in zip(coeffs, reversed(ds)))
    return x


def sample_dpm(model, x, sigmas, **solver_args):
    noise_schedule = NoiseScheduleKarras(sigmas)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    dpm_solver = DpmSolver(model, noise_schedule, algorithm_type="dpmsolver")
    x = dpm_solver.sample(
        x,
        steps=len(sigmas),
        t_start=sigma_max,
        t_end=sigma_min,
        **solver_args,
    )
    return x


def sample_dpmpp(model, x, sigmas, **solver_args):
    noise_schedule = NoiseScheduleKarras(sigmas)
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    dpm_solver = DpmSolver(model, noise_schedule, algorithm_type="dpmsolver++")
    x = dpm_solver.sample(
        x,
        steps=len(sigmas),
        t_start=sigma_max,
        t_end=sigma_min,
        **solver_args,
    )
    return x


def sample_sde_dpmpp(
    model, x, sigmas, callback=None, progress=False, order=2, algorithm_type="dpmsolver++", method="multistep"
):
    assert algorithm_type == "dpmsolver++", "Only dpmsolver++ is supported"
    assert method == "multistep", "Only multistep is supported"

    if order == 1:
        return sample_dpmpp_sde(model, x, sigmas, callback, progress)
    elif order == 2:
        return sample_dpmpp_2m_sde(model, x, sigmas, callback, progress)
    elif order == 3:
        return sample_dpmpp_3m_sde(model, x, sigmas, callback, progress)
    else:
        raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))


@torch.no_grad()
def sample_singlestep(
    denoiser,
    x,
    sigmas,
    progress=False,
):
    s_in = x.new_ones([x.shape[0]])
    return denoiser(x, sigmas[0] * s_in)


@torch.no_grad()
def sample_multistep(
    denoiser,
    x,
    sigmas,
    progress=False,
    sigma_min=0.002,
    sigma_max=80.0,
):
    """Algorithm 1 from https://arxiv.org/abs/2303.01469"""
    s_in = x.new_ones([x.shape[0]])

    for i in trange(len(sigmas) - 1, disable=not progress):
        x_0 = denoiser(x, sigmas[i] * s_in)
        sigma_next = torch.clip(sigmas[i + 1], sigma_min, sigma_max)
        z = torch.randn_like(x_0)
        x = x_0 + torch.sqrt(sigma_next**2 - sigma_min**2) * z

    return x
