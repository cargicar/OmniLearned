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
def sampler(model, X, y, num_steps, num_points, model_kwargs, device = "cuda", cond=None, pid=None, add_info=None):
    """
    Samples a clean point cloud from random Gaussian noise.

    Args:
        model: The trained diffusion model.
        num_steps: The number of denoising steps.
        Z: A random Gaussian noise tensor of shape [batch_size, num_points, 3].
        y: A tensor of shape [batch_size, num_classes] for one-hot encoding.
        cond: An optional tensor for conditional information.
        pid: An optional tensor for point cloud ID.
        add_info: Optional additional information tensor.

    Returns:
        The denoised point cloud tensor.
    """
    
    with torch.no_grad():

        #x_T = torch.randn([batch_size, num_points]).to(context.device)
        x = torch.randn_like(X)
        #traj = {self.var_sched.num_steps: x_T}# Start with the input noise
        # We sample from a high time (e.g., 1.0) down to a low time (e.g., epsilon)
        batch_size= x.shape[0]
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1).to(device)
        
        for time_step in torch.arange(num_steps, 0, -1):
            t = torch.ones((batch_size, 1)).to(x.device) * time_step / num_steps
            t_prev = torch.ones((batch_size, 1)).to(x.device) * (time_step - 1) / num_steps

            # Get logsnr, alpha, and sigma for current and previous timesteps
            logsnr_t, alpha, sigma =get_logsnr_alpha_sigma(t, shape=const_shape)
            logsnr_s, alpha_s, sigma_s = get_logsnr_alpha_sigma(t_prev, shape=const_shape)

            v = model(x, t, cond)

            # Predict the noise-free data (x_0)
            pred_x = alpha * x - sigma * v

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
    #     for i in range(num_steps):
    #         # Get current time and previous time
    #         t_current = timesteps[i]
    #         t_prev = timesteps[i+1]
            
    #         # Reshape time for model input
    #         t_current_tensor = torch.full((x.shape[0],), t_current).to(device)
    #         t_prev_tensor = torch.full((x.shape[0],), t_prev).to(device)
    #         # Get alpha and sigma for the current and previous timesteps
    #         _, alpha_current, sigma_current = get_logsnr_alpha_sigma(t_current_tensor)
    #         _, alpha_prev, sigma_prev = get_logsnr_alpha_sigma(t_prev_tensor)

    #         # Predict the velocity
    #         # The model's body takes the noisy data and conditions
    #         z_body = model.module.body(x, cond, pid, add_info, t_current_tensor)
    #         # The generator predicts the velocity
    #         z_pred_v = model.module.generator(z_body, y)
    #         #output_dic = model(x,y, **model_kwargs) # Doing from the whole model is weird. Does not take time?  
    #         #z_pred_v = output_dic["z_pred"]
    #         # --- Denoising Step ---
    #         # This is a simplified reverse step using a velocity-based update.
    #         # You can think of the predicted velocity as a direction to move.
    #         # The update rule needs to be derived from your specific diffusion process.
    #         # A common DDIM-style update might look like this:
            
    #         # Predict the "clean" data from the noisy data and predicted velocity
    #         # A clean-data estimate 'x0_hat' can be derived from the velocity
    #         # Assuming `v = - (sigma/alpha) * eps`, and `z = alpha*x + sigma*eps`
    #         # We can find `eps` and then `x`
    #         eps_pred = - (alpha_current / sigma_current) * z_pred_v
    #         x0_hat = (x - sigma_current * eps_pred) / alpha_current
            
    #         # Update the point cloud for the next step
    #         # The next state is a combination of the clean data estimate and noise
    #         x = alpha_prev * x0_hat + sigma_prev * eps_pred
    #         breakpoint()
    # return x

