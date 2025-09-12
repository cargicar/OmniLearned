"""
DPM-Solver (https://arxiv.org/abs/2206.00927) and DPM-Solver++ (https://arxiv.org/abs/2211.01095).
Based on the: https://github.com/LuChengTHU/dpm-solver

Script taken from https://arxiv.org/pdf/2509.07700
"""
import torch
import torchsde
from tqdm import trange

from src.utils import append_dims, append_zero


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0, device="cpu"):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n).to(device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


class NoiseScheduleKarras:
    def __init__(self, sigmas):
        self.sigmas = sigmas
        self.T = max(sigmas)
        self.total_N = len(sigmas)

    def log_alpha(self, t):
        return torch.zeros_like(t)

    def alpha(self, t):
        return torch.ones_like(t)

    def sigma(self, t):
        return t

    def lambda_(self, t):
        return -torch.log(t)

    def inverse_lambda(self, lamb):
        return torch.exp(-lamb)


class DpmSolver:
    def __init__(
        self,
        denoiser,
        noise_schedule,
        algorithm_type="dpmsolver++",
    ):
        """Construct a DPM-Solver.

        We support both DPM-Solver (`algorithm_type="dpmsolver"`) and DPM-Solver++ (`algorithm_type="dpmsolver++"`).

        Args:
            model_fn: A noise prediction model function which accepts the continuous-time input (t in [epsilon, T]):
                ``
                def model_fn(x, t_continuous):
                    return noise
                ``
                The shape of `x` is `(batch_size, **shape)`, and the shape of `t_continuous` is `(batch_size,)`.
            algorithm_type: A `str`. Either "dpmsolver" or "dpmsolver++".

        """
        self.model = denoiser
        self.noise_schedule = noise_schedule
        assert algorithm_type in ["dpmsolver", "dpmsolver++"]
        self.algorithm_type = algorithm_type

    def noise_prediction_fn(self, x, t):
        x0 = self.data_prediction_fn(x, t)
        alpha_t, sigma_t = self.noise_schedule.alpha(t), self.noise_schedule.sigma(t)
        alpha_t, sigma_t = append_dims(alpha_t, x.ndim), append_dims(sigma_t, x.ndim)
        noise = (x - alpha_t * x0) / sigma_t
        return noise

    def data_prediction_fn(self, x, t):
        return self.model(x, t)

    def model_fn(self, x, t):
        """
        Convert the model to the noise prediction model or the data prediction model.
        """
        t = t.expand(x.shape[0])
        if self.algorithm_type == "dpmsolver++":
            return self.data_prediction_fn(x, t)
        else:
            return self.noise_prediction_fn(x, t)

    def get_time_steps(self, skip_type, t_T, t_0, N, device):
        """Compute the intermediate time steps for sampling.

        Args:
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
                - 'karras': original karras time steps (Not tested yet).
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            N: A `int`. The total number of the spacing of the time steps.
            device: A torch device.
        Returns:
            A pytorch tensor of the time steps, with the shape (N + 1,).
        """
        if skip_type == "logSNR":
            lambda_T = self.noise_schedule.lambda_(torch.tensor(t_T).to(device))
            lambda_0 = self.noise_schedule.lambda_(torch.tensor(t_0).to(device))
            logSNR_steps = torch.linspace(lambda_T.cpu().item(), lambda_0.cpu().item(), N + 1).to(device)
            return self.noise_schedule.inverse_lambda(logSNR_steps)
        elif skip_type == "time_uniform":
            return torch.linspace(t_T, t_0, N + 1).to(device)
        elif skip_type == "time_quadratic":
            t_order = 2
            t = torch.linspace(t_T ** (1.0 / t_order), t_0 ** (1.0 / t_order), N + 1).pow(t_order).to(device)
            return t
        elif skip_type == "karras":
            return get_sigmas_karras(N, t_0, t_T, device=device)  # for karras diffusion t and sigma are the same
        else:
            raise ValueError(
                "Unsupported skip_type {}, need to be 'logSNR' or 'time_uniform' or 'time_quadratic' or 'karras'".format(
                    skip_type
                )
            )

    def get_orders_and_timesteps_for_singlestep_solver(self, steps, order, skip_type, t_T, t_0, device):
        """
        Get the order of each step for sampling by the singlestep DPM-Solver.

        We combine both DPM-Solver-1,2,3 to use all the function evaluations, which is named as "DPM-Solver-fast".
        Given a fixed number of function evaluations by `steps`, the sampling procedure by DPM-Solver-fast is:
            - If order == 1:
                We take `steps` of DPM-Solver-1 (i.e. DDIM).
            - If order == 2:
                - Denote K = (steps // 2). We take K or (K + 1) intermediate time steps for sampling.
                - If steps % 2 == 0, we use K steps of DPM-Solver-2.
                - If steps % 2 == 1, we use K steps of DPM-Solver-2 and 1 step of DPM-Solver-1.
            - If order == 3:
                - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                - If steps % 3 == 0, we use (K - 2) steps of DPM-Solver-3, and 1 step of DPM-Solver-2 and 1 step of DPM-Solver-1.
                - If steps % 3 == 1, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-1.
                - If steps % 3 == 2, we use (K - 1) steps of DPM-Solver-3 and 1 step of DPM-Solver-2.

        ============================================
        Args:
            order: A `int`. The max order for the solver (2 or 3).
            steps: A `int`. The total number of function evaluations (NFE).
            skip_type: A `str`. The type for the spacing of the time steps. We support three types:
                - 'logSNR': uniform logSNR for the time steps.
                - 'time_uniform': uniform time for the time steps. (**Recommended for high-resolutional data**.)
                - 'time_quadratic': quadratic time for the time steps. (Used in DDIM for low-resolutional data.)
                - 'karras': original karras time steps (Not tested yet).
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            device: A torch device.
        Returns:
            orders: A list of the solver order of each step.
        """
        if order == 3:
            K = steps // 3 + 1
            if steps % 3 == 0:
                orders = [3,] * (K - 2) + [2, 1]
            elif steps % 3 == 1:
                orders = [3,] * (K - 1) + [1]
            else:
                orders = [3,] * (K - 1) + [2]
        elif order == 2:
            if steps % 2 == 0:
                K = steps // 2
                orders = [2,] * K
            else:
                K = steps // 2 + 1
                orders = [2,] * (K - 1) + [1]
        elif order == 1:
            K = steps
            orders = [1,] * steps
        else:
            raise ValueError("'order' must be '1' or '2' or '3'.")
        if skip_type == 'logSNR':
            # To reproduce the results in DPM-Solver paper
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, K, device)
        else:
            timesteps_outer = self.get_time_steps(skip_type, t_T, t_0, steps, device)[torch.cumsum(torch.tensor([0,] + orders), 0).to(device)]
        return timesteps_outer, orders

    def denoise_to_zero_fn(self, x, s):
        """
        Denoise at the final step, which is equivalent to solve the ODE from lambda_s to infty by first-order discretization.
        """
        return self.data_prediction_fn(x, s)

    def dpm_solver_first_update(self, x, s, t, model_s=None, return_intermediate=False):
        """
        DPM-Solver-1 (equivalent to DDIM) from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s`.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.lambda_(s), ns.lambda_(t)
        h = lambda_t - lambda_s
        log_alpha_s, log_alpha_t = ns.log_alpha(s), ns.log_alpha(t)
        sigma_s, sigma_t = ns.sigma(s), ns.sigma(t)
        alpha_t = torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = sigma_t / sigma_s * x - alpha_t * phi_1 * model_s
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t
        else:
            phi_1 = torch.expm1(h)
            if model_s is None:
                model_s = self.model_fn(x, s)
            x_t = torch.exp(log_alpha_t - log_alpha_s) * x - (sigma_t * phi_1) * model_s
            if return_intermediate:
                return x_t, {'model_s': model_s}
            else:
                return x_t

    def singlestep_dpm_solver_second_update(self, x, s, t, r1=0.5, model_s=None, return_intermediate=False, solver_type="dpmsolver"):
        """
        Singlestep solver DPM-Solver-2 from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the second-order solver.
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            return_intermediate: A `bool`. If true, also return the model value at time `s` and `s1` (the intermediate time).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ["dpmsolver", "taylor"]:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 0.5
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.lambda_(s), ns.lambda_(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        s1 = ns.inverse_lambda(lambda_s1)
        log_alpha_s, log_alpha_s1, log_alpha_t = ns.log_alpha(s), ns.log_alpha(s1), ns.log_alpha(t)
        sigma_s, sigma_s1, sigma_t = ns.sigma(s), ns.sigma(s1), ns.sigma(t)
        alpha_s1, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_1 = torch.expm1(-h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = (sigma_s1 / sigma_s) * x - (alpha_s1 * phi_11) * model_s
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == "dpmsolver":
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    - (0.5 / r1) * (alpha_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == "taylor":
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (1.0 / r1) * (alpha_t * (phi_1 / h + 1.0)) * (model_s1 - model_s)
                )
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_1 = torch.expm1(h)

            if model_s is None:
                model_s = self.model_fn(x, s)
            x_s1 = torch.exp(log_alpha_s1 - log_alpha_s) * x - (sigma_s1 * phi_11) * model_s
            model_s1 = self.model_fn(x_s1, s1)
            if solver_type == "dpmsolver":
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (0.5 / r1) * (sigma_t * phi_1) * (model_s1 - model_s)
                )
            elif solver_type == "taylor":
                x_t = (
                    torch.exp(log_alpha_t - log_alpha_s) * x
                    - (sigma_t * phi_1) * model_s
                    - (1.0 / r1) * (sigma_t * (phi_1 / h - 1.0)) * (model_s1 - model_s)
                )

        if return_intermediate:
            return x_t, {'model_s': model_s, 'model_s1': model_s1}
        else:
            return x_t

    def singlestep_dpm_solver_third_update(
        self, x, s, t, r1=1.0 / 3.0, r2=2.0 / 3.0, model_s=None, model_s1=None, solver_type="dpmsolver"
    ):
        """
        Singlestep solver DPM-Solver-3 from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            r1: A `float`. The hyperparameter of the third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
            model_s: A pytorch tensor. The model function evaluated at time `s`.
                If `model_s` is None, we evaluate the model by `x` and `s`; otherwise we directly use it.
            model_s1: A pytorch tensor. The model function evaluated at time `s1` (the intermediate time given by `r1`).
                If `model_s1` is None, we evaluate the model at `s1`; otherwise we directly use it.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ["dpmsolver", "taylor"]:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        if r1 is None:
            r1 = 1.0 / 3.0
        if r2 is None:
            r2 = 2.0 / 3.0
        ns = self.noise_schedule
        lambda_s, lambda_t = ns.lambda_(s), ns.lambda_(t)
        h = lambda_t - lambda_s
        lambda_s1 = lambda_s + r1 * h
        lambda_s2 = lambda_s + r2 * h
        s1 = ns.inverse_lambda(lambda_s1)
        s2 = ns.inverse_lambda(lambda_s2)
        log_alpha_s, log_alpha_s1, log_alpha_s2, log_alpha_t = (
            ns.log_alpha(s),
            ns.log_alpha(s1),
            ns.log_alpha(s2),
            ns.log_alpha(t),
        )
        sigma_s, sigma_s1, sigma_s2, sigma_t = ns.sigma(s), ns.sigma(s1), ns.sigma(s2), ns.sigma(t)
        alpha_s1, alpha_s2, alpha_t = torch.exp(log_alpha_s1), torch.exp(log_alpha_s2), torch.exp(log_alpha_t)

        if self.algorithm_type == "dpmsolver++":
            phi_11 = torch.expm1(-r1 * h)
            phi_12 = torch.expm1(-r2 * h)
            phi_1 = torch.expm1(-h)
            phi_22 = torch.expm1(-r2 * h) / (r2 * h) + 1.0
            phi_2 = phi_1 / h + 1.0
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (sigma_s1 / sigma_s) * x - (alpha_s1 * phi_11) * model_s
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = (
                (sigma_s2 / sigma_s) * x
                - (alpha_s2 * phi_12) * model_s
                + r2 / r1 * (alpha_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == "dpmsolver":
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (1.0 / r2) * (alpha_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == "taylor":
                D1_0 = (1.0 / r1) * (model_s1 - model_s)
                D1_1 = (1.0 / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2.0 * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    (sigma_t / sigma_s) * x
                    - (alpha_t * phi_1) * model_s
                    + (alpha_t * phi_2) * D1
                    - (alpha_t * phi_3) * D2
                )
        else:
            phi_11 = torch.expm1(r1 * h)
            phi_12 = torch.expm1(r2 * h)
            phi_1 = torch.expm1(h)
            phi_22 = torch.expm1(r2 * h) / (r2 * h) - 1.0
            phi_2 = phi_1 / h - 1.0
            phi_3 = phi_2 / h - 0.5

            if model_s is None:
                model_s = self.model_fn(x, s)
            if model_s1 is None:
                x_s1 = (torch.exp(log_alpha_s1 - log_alpha_s)) * x - (sigma_s1 * phi_11) * model_s
                model_s1 = self.model_fn(x_s1, s1)
            x_s2 = (
                (torch.exp(log_alpha_s2 - log_alpha_s)) * x
                - (sigma_s2 * phi_12) * model_s
                - r2 / r1 * (sigma_s2 * phi_22) * (model_s1 - model_s)
            )
            model_s2 = self.model_fn(x_s2, s2)
            if solver_type == "dpmsolver":
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_s)) * x
                    - (sigma_t * phi_1) * model_s
                    - (1.0 / r2) * (sigma_t * phi_2) * (model_s2 - model_s)
                )
            elif solver_type == "taylor":
                D1_0 = (1.0 / r1) * (model_s1 - model_s)
                D1_1 = (1.0 / r2) * (model_s2 - model_s)
                D1 = (r2 * D1_0 - r1 * D1_1) / (r2 - r1)
                D2 = 2.0 * (D1_1 - D1_0) / (r2 - r1)
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_s)) * x
                    - (sigma_t * phi_1) * model_s
                    - (sigma_t * phi_2) * D1
                    - (sigma_t * phi_3) * D2
                )

        return x_t

    def multistep_dpm_solver_second_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"):
        """
        Multistep solver DPM-Solver-2 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if solver_type not in ["dpmsolver", "taylor"]:
            raise ValueError("'solver_type' must be either 'dpmsolver' or 'taylor', got {}".format(solver_type))
        ns = self.noise_schedule
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        t_prev_1, t_prev_0 = t_prev_list[-2], t_prev_list[-1]
        lambda_prev_1, lambda_prev_0, lambda_t = ns.lambda_(t_prev_1), ns.lambda_(t_prev_0), ns.lambda_(t)
        log_alpha_prev_0, log_alpha_t = ns.log_alpha(t_prev_0), ns.log_alpha(t)
        sigma_prev_0, sigma_t = ns.sigma(t_prev_0), ns.sigma(t)
        alpha_t = torch.exp(log_alpha_t)

        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0 = h_0 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            if solver_type == "dpmsolver":
                x_t = (sigma_t / sigma_prev_0) * x - (alpha_t * phi_1) * model_prev_0 - 0.5 * (alpha_t * phi_1) * D1_0
            elif solver_type == "taylor":
                x_t = (
                    (sigma_t / sigma_prev_0) * x
                    - (alpha_t * phi_1) * model_prev_0
                    + (alpha_t * (phi_1 / h + 1.0)) * D1_0
                )
        else:
            phi_1 = torch.expm1(h)
            if solver_type == "dpmsolver":
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                    - (sigma_t * phi_1) * model_prev_0
                    - 0.5 * (sigma_t * phi_1) * D1_0
                )
            elif solver_type == "taylor":
                x_t = (
                    (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                    - (sigma_t * phi_1) * model_prev_0
                    - (sigma_t * (phi_1 / h - 1.0)) * D1_0
                )
        return x_t

    def multistep_dpm_solver_third_update(self, x, model_prev_list, t_prev_list, t, solver_type="dpmsolver"):
        """
        Multistep solver DPM-Solver-3 from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        ns = self.noise_schedule
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list
        t_prev_2, t_prev_1, t_prev_0 = t_prev_list
        lambda_prev_2, lambda_prev_1, lambda_prev_0, lambda_t = (
            ns.lambda_(t_prev_2),
            ns.lambda_(t_prev_1),
            ns.lambda_(t_prev_0),
            ns.lambda_(t),
        )
        log_alpha_prev_0, log_alpha_t = ns.log_alpha(t_prev_0), ns.log_alpha(t)
        sigma_prev_0, sigma_t = ns.sigma(t_prev_0), ns.sigma(t)
        alpha_t = torch.exp(log_alpha_t)

        h_1 = lambda_prev_1 - lambda_prev_2
        h_0 = lambda_prev_0 - lambda_prev_1
        h = lambda_t - lambda_prev_0
        r0, r1 = h_0 / h, h_1 / h
        D1_0 = (1.0 / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1.0 / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1.0 / (r0 + r1)) * (D1_0 - D1_1)
        if self.algorithm_type == "dpmsolver++":
            phi_1 = torch.expm1(-h)
            phi_2 = phi_1 / h + 1.0
            phi_3 = phi_2 / h - 0.5
            x_t = (
                (sigma_t / sigma_prev_0) * x
                - (alpha_t * phi_1) * model_prev_0
                + (alpha_t * phi_2) * D1
                - (alpha_t * phi_3) * D2
            )
        else:
            phi_1 = torch.expm1(h)
            phi_2 = phi_1 / h - 1.0
            phi_3 = phi_2 / h - 0.5
            x_t = (
                (torch.exp(log_alpha_t - log_alpha_prev_0)) * x
                - (sigma_t * phi_1) * model_prev_0
                - (sigma_t * phi_2) * D1
                - (sigma_t * phi_3) * D2
            )
        return x_t

    def singlestep_dpm_solver_update(self, x, s, t, order, solver_type="dpmsolver", r1=None, r2=None):
        """
        Singlestep DPM-Solver with the order `order` from time `s` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            s: A pytorch tensor. The starting time, with the shape (1,).
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
            r1: A `float`. The hyperparameter of the second-order or third-order solver.
            r2: A `float`. The hyperparameter of the third-order solver.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, s, t)
        elif order == 2:
            return self.singlestep_dpm_solver_second_update(x, s, t, solver_type=solver_type, r1=r1)
        elif order == 3:
            return self.singlestep_dpm_solver_third_update(x, s, t, solver_type=solver_type, r1=r1, r2=r2)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def multistep_dpm_solver_update(self, x, model_prev_list, t_prev_list, t, order, solver_type="dpmsolver"):
        """
        Multistep DPM-Solver with the order `order` from time `t_prev_list[-1]` to time `t`.

        Args:
            x: A pytorch tensor. The initial value at time `s`.
            model_prev_list: A list of pytorch tensor. The previous computed model values.
            t_prev_list: A list of pytorch tensor. The previous times, each time has the shape (1,)
            t: A pytorch tensor. The ending time, with the shape (1,).
            order: A `int`. The order of DPM-Solver. We only support order == 1 or 2 or 3.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_t: A pytorch tensor. The approximated solution at time `t`.
        """
        if order == 1:
            return self.dpm_solver_first_update(x, t_prev_list[-1], t, model_s=model_prev_list[-1])
        elif order == 2:
            return self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        elif order == 3:
            return self.multistep_dpm_solver_third_update(x, model_prev_list, t_prev_list, t, solver_type=solver_type)
        else:
            raise ValueError("Solver order must be 1 or 2 or 3, got {}".format(order))

    def dpm_solver_adaptive(
        self, x, order, t_T, t_0, h_init=0.05, atol=0.0078, rtol=0.05, theta=0.9, t_err=1e-5, solver_type="dpmsolver"
    ):
        """
        The adaptive step size solver based on singlestep DPM-Solver.

        Args:
            x: A pytorch tensor. The initial value at time `t_T`.
            order: A `int`. The (higher) order of the solver. We only support order == 2 or 3.
            t_T: A `float`. The starting time of the sampling (default is T).
            t_0: A `float`. The ending time of the sampling (default is epsilon).
            h_init: A `float`. The initial step size (for logSNR).
            atol: A `float`. The absolute tolerance of the solver. For image data, the default setting is 0.0078 ((x_max - x_min) / 256), followed [1].
            rtol: A `float`. The relative tolerance of the solver. The default setting is 0.05.
            theta: A `float`. The safety hyperparameter for adapting the step size. The default setting is 0.9, followed [1].
            t_err: A `float`. The tolerance for the time. We solve the diffusion ODE until the absolute error between the
                current time and `t_0` is less than `t_err`. The default setting is 1e-5.
            solver_type: either 'dpmsolver' or 'taylor'. The type for the high-order solvers.
                The type slightly impacts the performance. We recommend to use 'dpmsolver' type.
        Returns:
            x_0: A pytorch tensor. The approximated solution at time `t_0`.

        [1] A. Jolicoeur-Martineau, K. Li, R. Piché-Taillefer, T. Kachman, and I. Mitliagkas, "Gotta go fast when generating data with score-based models," arXiv preprint arXiv:2105.14080, 2021.
        """
        ns = self.noise_schedule
        s = t_T * torch.ones((1,)).to(x)
        lambda_s = ns.lambda_(s)
        lambda_0 = ns.lambda_(t_0 * torch.ones_like(s).to(x))
        h = h_init * torch.ones_like(s).to(x)
        x_prev = x
        nfe = 0
        if order == 2:
            r1 = 0.5
            lower_update = lambda x, s, t: self.dpm_solver_first_update(x, s, t, return_intermediate=True)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, solver_type=solver_type, **kwargs)
        elif order == 3:
            r1, r2 = 1.0 / 3.0, 2.0 / 3.0
            lower_update = lambda x, s, t: self.singlestep_dpm_solver_second_update(x, s, t, r1=r1, return_intermediate=True, solver_type=solver_type)
            higher_update = lambda x, s, t, **kwargs: self.singlestep_dpm_solver_third_update(x, s, t, r1=r1, r2=r2, solver_type=solver_type, **kwargs)
        else:
            raise ValueError("For adaptive step size solver, order must be 2 or 3, got {}".format(order))
        while torch.abs((s - t_0)).mean() > t_err:
            t = ns.inverse_lambda(lambda_s + h)
            x_lower, lower_noise_kwargs = lower_update(x, s, t)
            x_higher = higher_update(x, s, t, **lower_noise_kwargs)
            delta = torch.max(torch.ones_like(x).to(x) * atol, rtol * torch.max(torch.abs(x_lower), torch.abs(x_prev)))
            norm_fn = lambda v: torch.sqrt(torch.square(v.reshape((v.shape[0], -1))).mean(dim=-1, keepdim=True))
            E = norm_fn((x_higher - x_lower) / delta).max()
            if torch.all(E <= 1.0):
                x = x_higher
                s = t
                x_prev = x_lower
                lambda_s = ns.lambda_(s)
            h = torch.min(theta * h * torch.float_power(E, -1.0 / order).float(), lambda_0 - lambda_s)
            nfe += order
        return x

    def sample(
        self,
        x,
        steps=20,
        t_start=None,
        t_end=None,
        order=2,
        skip_type="time_uniform",
        method="multistep",
        lower_order_final=True,
        denoise_to_zero=False,
        solver_type="dpmsolver",
        atol=0.0078,
        rtol=0.05,
        progress=False
    ):
        """
        Compute the sample at time `t_end` by DPM-Solver, given the initial `x` at time `t_start`.

        =====================================================

        We support the following algorithms for both noise prediction model and data prediction model:
            - 'singlestep':
                Singlestep DPM-Solver (i.e. "DPM-Solver-fast" in the paper), which combines different orders of singlestep DPM-Solver.
                We combine all the singlestep solvers with order <= `order` to use up all the function evaluations (steps).
                The total number of function evaluations (NFE) == `steps`.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    - If `order` == 1:
                        - Denote K = steps. We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - Denote K = (steps // 2) + (steps % 2). We take K intermediate time steps for sampling.
                        - If steps % 2 == 0, we use K steps of singlestep DPM-Solver-2.
                        - If steps % 2 == 1, we use (K - 1) steps of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                    - If `order` == 3:
                        - Denote K = (steps // 3 + 1). We take K intermediate time steps for sampling.
                        - If steps % 3 == 0, we use (K - 2) steps of singlestep DPM-Solver-3, and 1 step of singlestep DPM-Solver-2 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 1, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of DPM-Solver-1.
                        - If steps % 3 == 2, we use (K - 1) steps of singlestep DPM-Solver-3 and 1 step of singlestep DPM-Solver-2.
            - 'multistep':
                Multistep DPM-Solver with the order of `order`. The total number of function evaluations (NFE) == `steps`.
                We initialize the first `order` values by lower order multistep solvers.
                Given a fixed NFE == `steps`, the sampling procedure is:
                    Denote K = steps.
                    - If `order` == 1:
                        - We use K steps of DPM-Solver-1 (i.e. DDIM).
                    - If `order` == 2:
                        - We firstly use 1 step of DPM-Solver-1, then use (K - 1) step of multistep DPM-Solver-2.
                    - If `order` == 3:
                        - We firstly use 1 step of DPM-Solver-1, then 1 step of multistep DPM-Solver-2, then (K - 2) step of multistep DPM-Solver-3.
            - 'singlestep_fixed':
                Fixed order singlestep DPM-Solver (i.e. DPM-Solver-1 or singlestep DPM-Solver-2 or singlestep DPM-Solver-3).
                We use singlestep DPM-Solver-`order` for `order`=1 or 2 or 3, with total [`steps` // `order`] * `order` NFE.
            - 'adaptive':
                Adaptive step size DPM-Solver (i.e. "DPM-Solver-12" and "DPM-Solver-23" in the paper).
                We ignore `steps` and use adaptive step size DPM-Solver with a higher order of `order`.
                You can adjust the absolute tolerance `atol` and the relative tolerance `rtol` to balance the computatation costs
                (NFE) and the sample quality.
                    - If `order` == 2, we use DPM-Solver-12 which combines DPM-Solver-1 and singlestep DPM-Solver-2.
                    - If `order` == 3, we use DPM-Solver-23 which combines singlestep DPM-Solver-2 and singlestep DPM-Solver-3.

        =====================================================

        Some advices for choosing the algorithm:
            - For **unconditional sampling** or **guided sampling with small guidance scale** by DPMs:
                Use singlestep DPM-Solver or DPM-Solver++ ("DPM-Solver-fast" in the paper) with `order = 3`.
                e.g., DPM-Solver:
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
                e.g., DPM-Solver++:
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=3,
                            skip_type='time_uniform', method='singlestep')
            - For **guided sampling with large guidance scale** by DPMs:
                Use multistep DPM-Solver with `algorithm_type="dpmsolver++"` and `order = 2`.
                e.g.
                    >>> dpm_solver = DPM_Solver(model_fn, noise_schedule, algorithm_type="dpmsolver++")
                    >>> x_sample = dpm_solver.sample(x, steps=steps, t_start=t_start, t_end=t_end, order=2,
                            skip_type='time_uniform', method='multistep')

        We support three types of `skip_type`:
            - 'logSNR': uniform logSNR for the time steps. **Recommended for low-resolutional images**
            - 'time_uniform': uniform time for the time steps. **Recommended for high-resolutional images**.
            - 'time_quadratic': quadratic time for the time steps.
            - 'karras': original karras time steps (Not tested yet).

        =====================================================
        Args:
            x: A pytorch tensor. The initial value at time `t_start`
                e.g. if `t_start` == T, then `x` is a sample from the standard normal distribution.
            steps: A `int`. The total number of function evaluations (NFE).
            t_start: A `float`. The starting time of the sampling.
                If `T` is None, we use self.noise_schedule.T (default is 1.0).
            t_end: A `float`. The ending time of the sampling.
                If `t_end` is None, we use 1. / self.noise_schedule.total_N.
                e.g. if total_N == 1000, we have `t_end` == 1e-3.
                For discrete-time DPMs:
                    - We recommend `t_end` == 1. / self.noise_schedule.total_N.
                For continuous-time DPMs:
                    - We recommend `t_end` == 1e-3 when `steps` <= 15; and `t_end` == 1e-4 when `steps` > 15.
            order: A `int`. The order of DPM-Solver.
            skip_type: A `str`. The type for the spacing of the time steps. 'time_uniform' or 'logSNR' or 'time_quadratic' or 'karras'.
            method: A `str`. The method for sampling. 'singlestep' or 'multistep' or 'singlestep_fixed' or 'adaptive'.
            denoise_to_zero: A `bool`. Whether to denoise to time 0 at the final step.
                Default is `False`. If `denoise_to_zero` is `True`, the total NFE is (`steps` + 1).

                This trick is firstly proposed by DDPM (https://arxiv.org/abs/2006.11239) and
                score_sde (https://arxiv.org/abs/2011.13456). Such trick can improve the FID
                for diffusion models sampling by diffusion SDEs for low-resolutional images
                (such as CIFAR-10). However, we observed that such trick does not matter for
                high-resolutional images. As it needs an additional NFE, we do not recommend
                it for high-resolutional images.
            lower_order_final: A `bool`. Whether to use lower order solvers at the final steps.
                Only valid for `method=multistep` and `steps < 15`. We empirically find that
                this trick is a key to stabilizing the sampling by DPM-Solver with very few steps
                (especially for steps <= 10). So we recommend to set it to be `True`.
            solver_type: A `str`. The taylor expansion type for the solver. `dpmsolver` or `taylor`. We recommend `dpmsolver`.
            atol: A `float`. The absolute tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            rtol: A `float`. The relative tolerance of the adaptive step size solver. Valid when `method` == 'adaptive'.
            progress: A `bool`. Whether to show the progress bar. Default is `False`.
        Returns:
            x_end: A pytorch tensor. The approximated solution at time `t_end`.

        """
        t_0 = 1.0 / self.noise_schedule.total_N if t_end is None else t_end
        t_T = self.noise_schedule.T if t_start is None else t_start
        assert (
            t_0 > 0 and t_T > 0
        ), "Time range needs to be greater than 0. For discrete-time DPMs, it needs to be in [1 / N, 1], where N is the length of betas array"
        device = x.device
        
        if progress:
            pbar = trange(steps)
            
            def update_progress_bar(func):
                def wrapper(*args, **kwargs):
                    result = func(*args, **kwargs)
                    pbar.update(1)
                    return result
                return wrapper
            
            self.model_fn = update_progress_bar(self.model_fn)

        with torch.no_grad():
            if method == "adaptive":
                x = self.dpm_solver_adaptive(
                    x, order=order, t_T=t_T, t_0=t_0, atol=atol, rtol=rtol, solver_type=solver_type
                )
            elif method == "multistep":
                assert steps >= order
                timesteps = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=steps, device=device)
                assert timesteps.shape[0] - 1 == steps
                # Init the initial values.
                step = 0
                t = timesteps[step]
                t_prev_list = [t]
                model_prev_list = [self.model_fn(x, t)]
                # Init the first `order` values by lower order multistep DPM-Solver.
                for step in range(1, order):
                    t = timesteps[step]
                    x = self.multistep_dpm_solver_update(
                        x, model_prev_list, t_prev_list, t, step, solver_type=solver_type
                    )
                    t_prev_list.append(t)
                    model_prev_list.append(self.model_fn(x, t))
                # Compute the remaining values by `order`-th order multistep DPM-Solver.
                for step in range(order, steps + 1):
                    t = timesteps[step]
                    # We only use lower order for steps < 10
                    if lower_order_final and steps < 10:
                        step_order = min(order, steps + 1 - step)
                    else:
                        step_order = order
                    x = self.multistep_dpm_solver_update(
                        x, model_prev_list, t_prev_list, t, step_order, solver_type=solver_type
                    )
                    for i in range(order - 1):
                        t_prev_list[i] = t_prev_list[i + 1]
                        model_prev_list[i] = model_prev_list[i + 1]
                    t_prev_list[-1] = t
                    # We do not need to evaluate the final model value.
                    if step < steps:
                        model_prev_list[-1] = self.model_fn(x, t)
            elif method in ["singlestep", "singlestep_fixed"]:
                if method == "singlestep":
                    timesteps_outer, orders = self.get_orders_and_timesteps_for_singlestep_solver(
                        steps=steps, order=order, skip_type=skip_type, t_T=t_T, t_0=t_0, device=device
                    )
                elif method == "singlestep_fixed":
                    K = steps // order
                    orders = [order,] * K
                    timesteps_outer = self.get_time_steps(skip_type=skip_type, t_T=t_T, t_0=t_0, N=K, device=device)
                for step, order in enumerate(orders):
                    s, t = timesteps_outer[step], timesteps_outer[step + 1]
                    timesteps_inner = self.get_time_steps(
                        skip_type=skip_type, t_T=s.item(), t_0=t.item(), N=order, device=device
                    )
                    lambda_inner = self.noise_schedule.lambda_(timesteps_inner)
                    h = lambda_inner[-1] - lambda_inner[0]
                    r1 = None if order <= 1 else (lambda_inner[1] - lambda_inner[0]) / h
                    r2 = None if order <= 2 else (lambda_inner[2] - lambda_inner[0]) / h
                    x = self.singlestep_dpm_solver_update(x, s, t, order, solver_type=solver_type, r1=r1, r2=r2)
            else:
                raise ValueError("Got wrong method {}".format(method))
            if denoise_to_zero:
                t = torch.ones((x.shape[0],)).to(device) * t_0
                x = self.denoise_to_zero_fn(x, t)

        return x


# From: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/sampling.py
class BatchedBrownianTree:
    """A wrapper around torchsde.BrownianTree that enables batches of entropy."""

    def __init__(self, x, t0, t1, seed=None, **kwargs):
        t0, t1, self.sign = self.sort(t0, t1)
        w0 = kwargs.get("w0", torch.zeros_like(x))
        if seed is None:
            seed = torch.randint(0, 2**63 - 1, []).item()
        self.batched = True
        try:
            assert len(seed) == x.shape[0]
            w0 = w0[0]
        except TypeError:
            seed = [seed]
            self.batched = False
        self.trees = [torchsde.BrownianTree(t0, w0, t1, entropy=s, **kwargs) for s in seed]

    @staticmethod
    def sort(a, b):
        return (a, b, 1) if a < b else (b, a, -1)

    def __call__(self, t0, t1):
        t0, t1, sign = self.sort(t0, t1)
        w = torch.stack([tree(t0, t1) for tree in self.trees]) * (self.sign * sign)
        return w if self.batched else w[0]


class BrownianTreeNoiseSampler:
    """A noise sampler backed by a torchsde.BrownianTree.

    Args:
        x (Tensor): The tensor whose shape, device and dtype to use to generate
            random samples.
        sigma_min (float): The low end of the valid interval.
        sigma_max (float): The high end of the valid interval.
        seed (int or List[int]): The random seed. If a list of seeds is
            supplied instead of a single integer, then the noise sampler will
            use one BrownianTree per batch item, each with its own seed.
        transform (callable): A function that maps sigma to the sampler's
            internal timestep.
    """

    def __init__(self, x, sigma_min, sigma_max, seed=None, transform=lambda x: x):
        self.transform = transform
        t0, t1 = self.transform(torch.as_tensor(sigma_min)), self.transform(torch.as_tensor(sigma_max))
        self.tree = BatchedBrownianTree(x, t0, t1, seed)

    def __call__(self, sigma, sigma_next):
        t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
        return self.tree(t0, t1) / (t1 - t0).abs().sqrt()


@torch.no_grad()
def sample_dpmpp_sde(
    model,
    x,
    sigmas,
    callback=None,
    progress=False,
):
    """SDE-DPM-Solver++1"""
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max)
    s_in = x.new_ones([x.shape[0]])
    lambda_fn = lambda sigma: -sigma.log()

    for i in trange(len(sigmas) - 1, disable=not progress):
        denoised = model(x, sigmas[i] * s_in)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
            h = lambda_t - lambda_s
            x = sigmas[i + 1] / sigmas[i] * (-h).exp() * x + (-2 * h).expm1().neg() * denoised
            z = noise_sampler(sigmas[i], sigmas[i + 1])
            x = x + sigmas[i + 1] * (-2 * h).expm1().neg().sqrt() * z

    return x


@torch.no_grad()
def sample_dpmpp_2m_sde(
    model,
    x,
    sigmas,
    callback=None,
    progress=False,
):
    """SDE-DPM-Solver++(2M)"""
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max)
    s_in = x.new_ones([x.shape[0]])
    lambda_fn = lambda sigma: -sigma.log()

    old_denoised = None
    h_last = None

    for i in trange(len(sigmas) - 1, disable=not progress):
        denoised = model(x, sigmas[i] * s_in)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            x = denoised
        else:
            lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
            h = lambda_t - lambda_s

            x = sigmas[i + 1] / sigmas[i] * (-h).exp() * x + (-2 * h).expm1().neg() * denoised

            if old_denoised is not None:
                r1 = h_last / h
                x = x + 0.5 * (-2 * h).expm1().neg() * (denoised - old_denoised) / r1

            z = noise_sampler(sigmas[i], sigmas[i + 1])
            x = x + sigmas[i + 1] * (-2 * h).expm1().neg().sqrt() * z

        old_denoised = denoised
        h_last = h
    return x


@torch.no_grad()
def sample_dpmpp_3m_sde(
    model,
    x,
    sigmas,
    callback=None,
    progress=False,
):
    """SDE-DPM-Solver++(3M)"""
    sigma_min, sigma_max = sigmas[sigmas > 0].min(), sigmas.max()
    noise_sampler = BrownianTreeNoiseSampler(x, sigma_min, sigma_max)
    s_in = x.new_ones([x.shape[0]])
    lambda_fn = lambda sigma: -sigma.log()

    denoised_1, denoised_2 = None, None
    h_1, h_2 = None, None

    for i in trange(len(sigmas) - 1, disable=not progress):
        denoised = model(x, sigmas[i] * s_in)
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        if sigmas[i + 1] == 0:
            # Denoising step
            x = denoised
        else:
            lambda_s, lambda_t = lambda_fn(sigmas[i]), lambda_fn(sigmas[i + 1])
            h = lambda_t - lambda_s

            x = torch.exp(-2 * h) * x + (-2 * h).expm1().neg() * denoised

            if h_2 is not None:
                r1 = h_1 / h
                r2 = h_2 / h
                d1_1 = (denoised - denoised_1) / r1
                d1_2 = (denoised_1 - denoised_2) / r2
                d1 = d1_1 + (d1_1 - d1_2) * r1 / (r1 + r2)
                d2 = (d1_1 - d1_2) / (r1 + r2)
                phi_2 = (-2 * h).expm1() / (-2 * h) + 1
                phi_3 = phi_2 / (-2 * h) - 0.5
                x = x + phi_2 * d1 - phi_3 * d2
            elif h_1 is not None:
                r = h_1 / h
                d = (denoised - denoised_1) / r
                phi_2 = (-2 * h).expm1() / (2 * h) + 1
                x = x + phi_2 * d

            z = noise_sampler(sigmas[i], sigmas[i + 1])
            x = x + sigmas[i + 1] * (-2 * h).expm1().neg().sqrt() * z

        denoised_1, denoised_2 = denoised, denoised_1
        h_1, h_2 = h, h_1
    return x