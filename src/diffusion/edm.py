"""
Implementation of Elucidating the Design Space of Diffusion-Based Generative Models (EDM) from: https://arxiv.org/abs/2206.00364.
Based on the: https://github.com/openai/consistency_models

Script taken from https://arxiv.org/pdf/2509.07700
"""

import torch

from src.diffusion.base import DiffusionModelBase
from src.diffusion.sampling import (
    sample_dpm,
    sample_dpmpp,
    sample_euler,
    sample_heun,
    sample_linear_multistep,
    sample_sde_dpmpp,
)
from src.utils import append_dims, append_zero, default, mean_flat

class EDM(DiffusionModelBase):
    def __init__(
        self,
        model,
        num_timesteps=40,
        sigma_min=0.002,  # min noise level
        sigma_max=80,  # max noise level
        sigma_data=0.5,  # standard deviation of data distribution
        rho=7,  # controls the sampling schedule
        P_mean=-1.2,  # mean of log-normal distribution from which noise is drawn for training
        P_std=1.2,  # standard deviation of log-normal distribution from which noise is drawn for training
    ):
        super().__init__()
        self.model = model
        #assert model.in_channels == model.out_channels, "input and output channels must match"
        #TODO figure out how to pass this info from the model
        #self.input_size = (model.in_channels, *model.input_size)

        self.num_timesteps = num_timesteps

        # noise level parameters
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data

        # sampling schedule parameters
        self.rho = rho

        # noise level parameters
        self.P_mean = P_mean
        self.P_std = P_std

    # network and preconditioning
    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def denoise(self, x_t, x_cond, sigma):
        c_skip, c_out, c_in = [append_dims(c, x_t.ndim) for c in self.get_scalings(sigma)]
        c_noise = 0.25 * torch.log(sigma + 1e-44) * 1000 #  see: https://github.com/openai/consistency_models/blob/main/cm/karras_diffusion.py#L346
        model_output = self.model(c_in * x_t, x_cond, c_noise)
        z_pred = model_output["z_pred"]
        #return c_skip * x_t + c_out * model_output
        return c_skip * x_t + c_out * z_pred

    # training
    def noise_distribution(self, batch_size):
        # log(sigma) ~ N(P_mean, P_std^2)
        log_sigmas = self.P_mean + self.P_std * torch.randn((batch_size,), device=self.device)
        return torch.exp(log_sigmas)

    def loss_weighting(self, sigma):
        # lambda(t)
        return (sigma**2 + self.sigma_data**2) * (sigma * self.sigma_data) ** -2

    def forward(self, x_0, x_cond):
        batch_size = x_0.shape[0]
        sigmas = self.noise_distribution(batch_size)
        noise = torch.randn_like(x_0)
        x_t = x_0 + noise * append_dims(sigmas, x_0.ndim)
        denoised = self.denoise(x_t, x_cond, sigmas)
        weights = append_dims(self.loss_weighting(sigmas), x_0.ndim)
        losses = mean_flat(weights * (denoised - x_0) ** 2)
        return losses.mean()

    # sampling
    def get_timesteps(self, steps=None):
        N = default(steps, self.num_timesteps)
        i = torch.arange(N, device=self.device)
        sigmas = (
            self.sigma_max ** (1 / self.rho)
            + i / (N - 1) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
        ) ** self.rho
        return append_zero(sigmas)

    def get_sigmas(self, t):
        # sigma(t) = t
        return t

    def get_snrs(self, t):
        return t**-2

    @torch.inference_mode()
    def sample(
        self,
        x_shape, 
        #conditions,
        conditions, #conditions
        steps=None,
        solver="heun",
        solver_args={},
        clip_denoised=False,
        progress=False,
    ):
        #x_shape = (num_samples, *self.input_size)
        
        num_samples = len(conditions[0])
        assert all(num_samples == len(c) for c in conditions), "all conditions must have the same batch size"

        ts = self.get_timesteps(steps)
        sigmas = self.get_sigmas(ts)

        x_T = torch.randn(x_shape, device=self.device) * self.sigma_max

        sample_fn = {
            "heun": sample_heun,
            "euler": sample_euler,
            "linear_multistep": sample_linear_multistep,
            "dpmsolver": sample_dpm,
            "dpmsolver++": sample_dpmpp,
            "sde-dpmsolver++": sample_sde_dpmpp,
        }[solver]

        def denoiser(x_t, sigma):
            denoised = self.denoise(x_t, conditions, sigma)
            if clip_denoised:
                denoised = denoised.clamp(-1, 1)
            return denoised

        x_0 = sample_fn(
            denoiser,
            x_T,
            sigmas,
            progress=progress,
            **solver_args,
        )
        return x_0
