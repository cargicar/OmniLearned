import torch
import torch.nn as nn
import numpy as np


class MPFourier(nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer("freqs", 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer("phases", 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        return x.unsqueeze(-1) * y.to(x.dtype)


@torch.compile
def logsnr_schedule_cosine(t, logsnr_min=-20.0, logsnr_max=20.0, shift=1.0):
    b = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max)))
    a = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min))) - b
    return -2.0 * torch.log(torch.tan(a * t + b) * shift)


# @torch.compile
def get_logsnr_alpha_sigma(time, shift=1.0):
    logsnr = logsnr_schedule_cosine(time, shift=shift)[:, None, None]
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))
    return logsnr, alpha, sigma


def perturb(x, time):
    #mask = x[:, :, 3:4] != 0
    mask = x[:, :, 2:3] != 0
    eps = torch.randn_like(x)  # eps ~ N(0, 1)
    logsnr, alpha, sigma = get_logsnr_alpha_sigma(time)
    z = alpha * x + eps * sigma
    v = alpha * eps - sigma * x
    return z * mask, v * mask



########################################################################
#################### Added for Shapenet branch #########################
#TODO this should be move to network #################3
# --- Sampler Function ---
def DDIM_sampler(model, X, y, gap, energy, num_steps= 1000, num_points= 500, device = "cuda", cond=None, pid=None, add_info=None):
    """
    Samples a clean point cloud from random Gaussian noise.

    Args:
        model: The trained diffusion model.
        num_steps: The number of denoising steps.
        X: A random Gaussian noise tensor of shape [batch_size, num_points, 4].
        y: A tensor of shape [batch_size, particle class] .
        gap: A tensor of shape [batch_size, gap class] .
        energy: A tensor of shape [batch_size, energy primary (normalized)].

    Returns:
        The denoised point cloud tensor.
    """
    
    with torch.no_grad():

        #x_T = torch.randn([batch_size, num_points]).to(context.device)
        x = torch.randn_like(X)
        #traj = {self.var_sched.num_steps: x_T}# Start with the input noise
        # We sample from a high time (e.g., 1.0) down to a low time (e.g., epsilon)
        data_shape = x.shape
        batch_size= x.shape[0]
        #timesteps = torch.linspace(1.0, 0.0, num_steps + 1).to(device)
        
        for time_step in torch.arange(num_steps, 0, -1):
            #time = torch.rand(size=(x.shape[0],)).to(x.device) #training time
            t = torch.ones((batch_size, )).to(x.device) * time_step / num_steps
            t_prev = torch.ones((batch_size, )).to(x.device) * (time_step - 1) / num_steps

            # Get logsnr, alpha, and sigma for current and previous timesteps
            logsnr_t, alpha, sigma =get_logsnr_alpha_sigma(t)
            logsnr_s, alpha_s, sigma_s = get_logsnr_alpha_sigma(t_prev)
            #v = model(x, t, cond)
            # Predict the velocity
            # The model's body takes the noisy data and conditions
            # TODO, I can read the whole batch here, and pass the whole model instead of fraction it in body and generator
            z_body = model.body(x, cond, pid, add_info, t)
            # The generator predicts the velocity
            z_pred_v = model.generator(z_body, y, gap, energy)
            #output_dic = model(x,y, **model_kwargs) # Doing from the whole model is weird. Does not take time?  
            #z_pred_v = output_dic["z_pred"]
            # --- Denoising Step ---
            # Predict the noise-free data (x_0)
            pred_x = alpha * x - sigma * z_pred_v

            # Calculate the mean and standard deviation for the next step (x_{t-1})
            alpha_st = torch.sqrt((1. + torch.exp(-logsnr_t)) / (1. + torch.exp(-logsnr_s)))
            r = torch.exp(logsnr_t - logsnr_s)  # SNR(t)/SNR(s)
            one_minus_r = -torch.expm1(logsnr_t - logsnr_s)  # 1-SNR(t)/SNR(s)

            mean = r * alpha_st * x + one_minus_r * alpha_s * pred_x
            std = torch.sqrt(one_minus_r) * sigma_s

            # Generate random noise for the next step
            eps = torch.randn(data_shape, dtype=torch.float32).to(x.device)

            # Update x for the next timestep
            x = mean + std * eps
        
        return pred_x

#FIXME it need lo look more like the Euler Sampler from RF repo. Like in evaluate_RF.py
def RF_sampler(model, X, y, gap, energy, num_steps= 1000, num_points= 500, device = "cuda", cond=None, pid=None, add_info=None):
    """
    Samples a clean point cloud from random Gaussian noise.

    Args:
        model: The trained diffusion model.
        num_steps: The number of denoising steps.
        X: A random Gaussian noise tensor of shape [batch_size, num_points, 4].
        y: A tensor of shape [batch_size, particle class] .
        gap: A tensor of shape [batch_size, gap class] .
        energy: A tensor of shape [batch_size, energy primary (normalized)].

    Returns:
        The denoised point cloud tensor.
    """
    
    with torch.no_grad():

        #x_T = torch.randn([batch_size, num_points]).to(context.device)
        x = torch.randn_like(X)#.to(device)
        # Discretize the time from 0 to 1
        dt = 1.0 / num_steps
        #times = torch.arange(0, 1, dt)#.to(device)
        #traj = {self.var_sched.num_steps: x_T}# Start with the input noise
        # We sample from a high time (e.g., 1.0) down to a low time (e.g., epsilon)
        data_shape = x.shape
        batch_size= x.shape[0]
        #timesteps = torch.linspace(1.0, 0.0, num_steps + 1).to(device)
        
        #for time_step in times:
        for time_step in torch.arange(num_steps, 0, -1):
            #time = torch.rand(size=(x.shape[0],)).to(x.device) #training time
            #t = time_step.repeat(x.shape[0])[:, None] # Adjust shape for model input
            t = torch.ones((batch_size, )).to(x.device) * time_step / num_steps
        
            # Predict the velocity field with your model
            # The model's body takes the noisy data and conditions
            z_body = model.body(x, cond, pid, add_info, t)
            # The generator predicts the velocity
            v = model.generator(z_body, y, gap, energy)
            
            # Euler integration step
            x = x + v * dt
        
        return x
    
import torch
import random
import numpy as np

import matplotlib.pyplot as plt

from rectified_flow.rectified_flow import RectifiedFlow
from rectified_flow.utils import set_seed, match_dim_with_data
from typing import Callable


class Sampler:
    r"""Sampler class for generating samples using Rectified Flow models.

    This class provides a general framework for sampling from a Rectified Flow model over a specified time grid.
    The sampling process starts from initial samples drawn from the source distribution `pi_0` at time `t = 0` and evolves
    these samples over time towards the target distribution at time `t = 1`.

    At each time step, the state `X_t` is updated according to the model's velocity field, which depends on both the current
    state and time. The sampling process proceeds by iteratively updating `X_t` and advancing the time `t` along the specified
    time grid.

    The `Sampler` class is designed to be subclassed, with the `step` method implemented to define the integration scheme
    used to update `x_t`.

    To create a custom sampler, subclass `Sampler` and implement the `step` method. The `step` method should update `self.x_t`
    based on the current state, time, and any model-specific arguments.

    **Example:**

    ```python
    class EulerSampler(Sampler):
        def step(self, **model_kwargs):
            dt = self.t_next - self.t
            v_t = self.get_velocity(**model_kwargs)
            self.x_t = self.x_t + v_t * dt

    # Create a RectifiedFlow instance (not shown here)
    rectified_flow = RectifiedFlow(...)

    # Initialize the sampler
    sampler = EulerSampler(rectified_flow, num_steps=100)

    # Run the sampling loop
    sampler.sample_loop(num_samples=1000)
    ```

    In this example, `EulerSampler` implements a simple Euler integration scheme for the sampling process.
    """

    def __init__(  # NOTE: consider using dataclass config
        self,
        rectified_flow: RectifiedFlow,
        num_steps: int | None = None,
        time_grid: list[float] | torch.Tensor | None = None,
        record_traj_period: int = 1,
        callbacks: list[Callable] | None = None,
        num_samples: int | None = None,
    ):
        r"""Initialize the Sampler instance.

        Args:
            rectified_flow (`RectifiedFlow`):
                The RectifiedFlow instance to sample from.
            num_steps (`int`, *optional*):
                Number of time steps for sampling. If `time_grid` is not provided, `num_steps` must be specified to generate the time grid.
            time_grid (`list[float]` or `torch.Tensor`, *optional*):
                Time grid for sampling. A list or tensor specifying the time points between 0 and 1 (inclusive). If not provided, it will be generated uniformly based on `num_steps`.
            record_traj_period (`int`, *optional*):
                Period at which to record the trajectory during sampling. Defaults to `1`, recording every step.
            callbacks (`list[Callable]`, *optional*):
                List of callback functions to be called at each recording step. Each function should take the sampler instance as an argument.
            num_samples (`int`, *optional*):
                Number of samples to generate. If not provided here, it must be specified when calling `sample_loop`.
        """
        self.rectified_flow = rectified_flow

        # Prepare time grid
        if num_steps is not None or time_grid is not None:
            self.num_steps, self.time_grid = self._prepare_time_grid(
                num_steps, time_grid
            )
        else:
            self.num_steps = None
            self.time_grid = None

        self.callbacks = callbacks or []
        self.record_traj_period = record_traj_period

        # Initialize sampling state
        self.num_samples = num_samples
        self.x_t = None
        self.x_0 = None
        self.step_count = 0

    def _prepare_time_grid(self, num_steps, time_grid):
        r"""Prepare the time grid for sampling.

        Args:
            num_steps (`int`):
                Number of steps to divide the interval `[0, 1]` into. If `time_grid` is not provided, a uniform grid is created with `num_steps + 1` points.
            time_grid (`list[float]` or `torch.Tensor`):
                A list or tensor of time points between `0` and `1` (inclusive). If provided, it must have `num_steps + 1` elements if `num_steps` is also specified.
        """
        if num_steps is None and time_grid is None:
            raise ValueError("At least one of num_steps or time_grid must be provided")

        if time_grid is None:
            time_grid = np.linspace(0, 1, num_steps + 1).tolist()
        else:
            if isinstance(time_grid, torch.Tensor):
                time_grid = time_grid.tolist()
            elif isinstance(time_grid, np.ndarray):
                time_grid = time_grid.tolist()
            elif not isinstance(time_grid, list):
                time_grid = list(time_grid)

            if num_steps is None:
                num_steps = len(time_grid) - 1
            else:
                assert (
                    len(time_grid) == num_steps + 1
                ), "Time grid must have num_steps + 1 elements"

        return num_steps, time_grid

    def step(self, **model_kwargs):
        r"""Perform a single integration step.
        This method should be overridden by subclasses to implement specific integration schemes.
        """
        raise NotImplementedError("Subclasses should implement this step method.")

    def set_next_time_point(self):
        r"""Advance to the next time point in the time grid.
        This method updates the current time `t` and the next time `t_next` by advancing the iterator over the time grid.
        """
        self.step_count += 1
        try:
            self.t = self.t_next
            self.t_next = next(self.time_iter)
        except StopIteration:
            self.t_next = None

    def stop(self):
        r"""Determine whether the sampling loop should stop.
        The sampling stops when there are no more time points in the time grid or when the current time `t` reaches the final time point (typically `1.0`).
        """
        return self.t_next is None or self.t >= 1.0 - 1e-6

    def record(self):
        r"""Record the current state of the sampling process.
        This method records the current state `x_t` and time `t` at the specified recording period. It also executes any callbacks provided.
        """
        if self.step_count % self.record_traj_period == 0:
            self._trajectories.append(self.x_t.detach().clone())
            self._time_points.append(self.t)

            # Callbacks can be used for logging or additional recording
            for callback in self.callbacks:
                callback(self)

    @torch.no_grad()
    def sample_loop(
        self,
        num_samples: int | None = None,
        x_0: torch.Tensor | None = None,
        seed: int | None = None,
        num_steps: int | None = None,
        time_grid: list[float] | torch.Tensor | None = None,
        **model_kwargs,
    ):
        r"""Run the sampling loop to generate samples.

        This method performs the sampling by iteratively calling the `step` method, recording trajectories, and updating the state.

        Args:
            num_samples (`int`, *optional*):
                Number of samples to generate. If not provided, it must be specified in the constructor or inferred from `x_0`.
            x_0 (`torch.Tensor`, *optional*):
                Initial samples from the source distribution `pi_0`. If not provided, samples are drawn from `rectified_flow.sample_source_distribution`.
            seed (`int`, *optional*):
                Random seed for reproducibility.
            num_steps (`int`, *optional*):
                Number of time steps for sampling. If provided, overrides the `num_steps` provided during initialization.
            time_grid (`list[float]` or `torch.Tensor`, *optional*):
                Time grid for sampling. If provided, overrides, overrides the `time_grid` provided during initialization.
            **model_kwargs:
                Additional keyword arguments to pass to the velocity field model.

        Returns:
            `Sampler`:
                The sampler instance with the sampling results.
        """
        if seed is not None:
            set_seed(seed)

        if num_samples is None:
            if x_0 is not None:
                num_samples = x_0.shape[0]
            elif self.num_samples is not None:
                num_samples = self.num_samples
            else:
                raise ValueError(
                    "num_samples must be specified if x_0 is not provided."
                )
        self.num_samples = num_samples

        # Prepare initial state
        if x_0 is not None:
            self.x_t = x_0.clone()
        else:
            self.x_t = self.rectified_flow.sample_source_distribution(num_samples)
        self.x_0 = self.x_t.clone()

        # Prepare time grid, can be overridden when calling the method
        if num_steps is not None or time_grid is not None:
            self.num_steps, self.time_grid = self._prepare_time_grid(
                num_steps, time_grid
            )

        self.step_count = 0
        self.time_iter = iter(self.time_grid)
        self.t = next(self.time_iter)
        self.t_next = next(self.time_iter)

        # Recording trajectories
        self._trajectories = [self.x_t.detach().clone()]
        self._time_points = [self.t]

        # Runs the sampling process
        while not self.stop():
            self.step(**model_kwargs)
            self.record()
            self.set_next_time_point()

        return self

    @property
    def trajectories(self) -> list[torch.Tensor]:
        """List of recorded trajectories."""
        return self._trajectories        


    @property
    def time_points(self) -> list[float]:
        """List of recorded time points."""
        return self._time_points

