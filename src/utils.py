import importlib
import logging
import math
import os
import random
import sys
from functools import wraps
from typing import Union, Tuple, List

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
#from accelerate import Accelerator, DataLoaderConfiguration, DistributedDataParallelKwargs
#from omegaconf import OmegaConf
from sklearn import metrics
import torch.nn as nn
from torch.distributed import init_process_group, get_rank
import torch.nn.functional as F


##### omnilearned utils #####

def print_metrics(y_preds_np, y_np, thresholds=[0.3, 0.5], background_class=0):
    # Compute multiclass AUC
    auc_ovo = metrics.roc_auc_score(
        y_np,
        y_preds_np if y_preds_np.shape[-1] > 2 else y_preds_np[:, -1],
        multi_class="ovo",
    )
    print(f"AUC: {auc_ovo:.4f}\n")

    num_classes = y_preds_np.shape[1]

    for signal_class in range(num_classes):
        if signal_class == background_class:
            continue

        # Create binary labels: 1 for signal_class, 0 for background_class, ignore others
        mask = (y_np == signal_class) | (y_np == background_class)
        y_bin = (y_np[mask] == signal_class).astype(int)
        scores_bin = y_preds_np[mask, signal_class] / (
            y_preds_np[mask, signal_class] + y_preds_np[mask, background_class]
        )

        # Compute ROC
        fpr, tpr, _ = metrics.roc_curve(y_bin, scores_bin)

        print(f"Signal class {signal_class} vs Background class {background_class}:")

        for threshold in thresholds:
            bineff = np.argmax(tpr > threshold)
            print(
                "Class {} effS at {} 1.0/effB = {}".format(
                    signal_class, tpr[bineff], 1.0 / fpr[bineff]
                )
            )


class CLIPLoss(nn.Module):
    # From AstroCLIP: https://github.com/PolymathicAI/AstroCLIP/blob/main/astroclip/models/astroclip.py#L117
    def get_logits(
        self,
        clean_features: torch.FloatTensor,
        perturbed_features: torch.FloatTensor,
        logit_scale: float,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        # Normalize image features
        clean_features = F.normalize(clean_features, dim=-1, eps=1e-3)

        # Normalize spectrum features
        perturbed_features = F.normalize(perturbed_features, dim=-1, eps=1e-3)

        # Calculate the logits for the image and spectrum features

        logits_per_clean = logit_scale * clean_features @ perturbed_features.T
        return logits_per_clean, logits_per_clean.T

    def forward(
        self,
        clean_features: torch.FloatTensor,
        perturbed_features: torch.FloatTensor,
        weight=None,
        logit_scale: float = 2.74,
        output_dict: bool = False,
    ) -> torch.FloatTensor:
        # Get the logits for the clean and perturbed features
        logits_per_clean, logits_per_perturbed = self.get_logits(
            clean_features, perturbed_features, logit_scale
        )

        # Calculate the contrastive loss
        labels = torch.arange(
            logits_per_clean.shape[0], device=clean_features.device, dtype=torch.long
        )
        total_loss = (
            F.cross_entropy(logits_per_clean, labels, reduction="none")
            + F.cross_entropy(logits_per_perturbed, labels, reduction="none")
        ) / 2
        if weight is not None:
            total_loss = torch.mean(weight * total_loss)
        else:
            total_loss = total_loss.mean()
        return {"contrastive_loss": total_loss} if output_dict else total_loss


def sum_reduce(num, device):
    r"""Sum the tensor across the devices."""
    if not torch.is_tensor(num):
        rt = torch.tensor(num).to(device)
    else:
        rt = num.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def get_param_groups(model, wd, lr, lr_factor=1.0, fine_tune=False):
    no_decay, decay = [], []
    last_layer_no_decay, last_layer_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_last_layer = name.startswith("classifier.out")

        if any(keyword in name for keyword in model.no_weight_decay()):
            if is_last_layer:
                last_layer_no_decay.append(param)
            else:
                no_decay.append(param)
        else:
            if is_last_layer:
                last_layer_decay.append(param)
            else:
                decay.append(param)

    # Base learning rate groups
    param_groups = [
        {"params": decay, "weight_decay": wd, "lr": lr},
        {"params": no_decay, "weight_decay": 0.0, "lr": lr},
    ]

    # Adjust learning rate for last layer if fine-tuning
    last_layer_lr = lr * lr_factor if fine_tune else lr

    if last_layer_decay:
        param_groups.append(
            {"params": last_layer_decay, "weight_decay": wd, "lr": last_layer_lr}
        )
    if last_layer_no_decay:
        param_groups.append(
            {"params": last_layer_no_decay, "weight_decay": 0.0, "lr": last_layer_lr}
        )

    return param_groups


def get_checkpoint_name(tag):
    return f"best_model_{tag}.pt"


def is_master_node():
    if "RANK" in os.environ:
        return int(os.environ["RANK"]) == 0
    else:
        return True


def ddp_setup():
    """
    Args:
        rank: Unique identifixer of each process
        world_size: Total number of processes
    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "2900"
        os.environ["RANK"] = "0"
        init_process_group(rank=0, world_size=1)
        rank = local_rank = 0
    else:
        init_process_group(init_method="env://")
        # overwrite variables with correct values from env
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = get_rank()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.benchmark = True

    return local_rank, rank, dist.get_world_size()

##### edm utils ##########################

def get_logger(name: str = None, zero_rank_only: bool = True):
    """Create a logger that writes to stdout"""
    is_distributed = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    if name is None:
        name = __name__

    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    if zero_rank_only and rank != 0:
        logger.addHandler(logging.NullHandler())  # no-op logger
    else:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

#TODO addepd Accelerator setup to DDp
# def setup_accelerator(cpu: bool = False, mixed_precision: str = "none", single_core: bool = False) -> Accelerator:
#     mp.set_start_method("spawn")

#     if cpu and single_core:
#         os.environ["OPENBLAS_NUM_THREADS"] = "1"
#         torch.set_num_threads(1)

#     # keep split_batches unchanged; resuming a run with different resources and split_batches=False can modify batches_per_epoch
#     # do not remove find_unused_parameters, it is necessary for DDP to work properly: https://github.com/pytorch/pytorch/issues/43259
#     accelerator = Accelerator(
#         cpu=cpu,
#         mixed_precision=mixed_precision,
#         dataloader_config=DataLoaderConfiguration(split_batches=True),
#         kwargs_handlers=[
#             DistributedDataParallelKwargs(find_unused_parameters=True),
#         ],
#     )

#     # if distributed, wait for all processes to join
#     if accelerator.use_distributed:
#         accelerator.wait_for_everyone()

#     return accelerator


def set_seed(seed: int, deterministic: bool = False, all_gpus: bool = False):
    """Set the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if all_gpus:
        torch.cuda.manual_seed_all(seed)
    # can slow down training
    if deterministic:
        torch.use_deterministic_algorithms(True)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cycle(dataloader):
    while True:
        for sample in dataloader:
            yield sample


def identity(x, *args, **kwargs):
    return x


def to_device(x, device: str = "cuda"):
    if isinstance(x, (list, tuple)):
        return tuple(to_device(item, device) for item in x)
    elif isinstance(x, dict):
        return {key: to_device(value, device) for key, value in x.items()}
    else:
        return x.to(device)


def import_class_by_name(class_name: str):
    module_name, class_name = class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


# def load_config(default_config_path: str):
#     """
#     Load config from default path, merge it with custom config (if provided) and CLI arguments.
#     Then call the decorated function with the config as an only argument.
#     """

#     def _is_yaml_file(file_path: str) -> bool:
#         """Check if the file is a YAML file."""
#         return file_path.endswith(".yaml") or file_path.endswith(".yml")

#     def decorator(func):
#         @wraps(func)
#         def wrapper():
#             config = OmegaConf.load(default_config_path)

#             args = sys.argv[1:]
#             if len(args) > 0 and _is_yaml_file(args[0]):
#                 custom_config_path = args.pop(0)
#                 print(custom_config_path)
#                 custom_config = OmegaConf.load(custom_config_path)
#                 config = OmegaConf.merge(config, custom_config)

#             if len(args) > 0:
#                 cli_config = OmegaConf.from_cli(args)
#                 config = OmegaConf.merge(config, cli_config)

#             func(config)

#         return wrapper

#     return decorator


def flatten_dict(nested_dict: dict, sep="."):
    """Flatten a nested dictionary into a single level dictionary."""

    def _flatten_dict(nested_dict, parent_key=""):
        items = []
        for key, value in nested_dict.items():
            new_key = parent_key + sep + key if parent_key else key
            if isinstance(value, dict):
                items.extend(_flatten_dict(value, new_key).items())
            else:
                items.append((new_key, value))
        return dict(items)

    return _flatten_dict(nested_dict)


def get_lrs(optimizer):
    return [param_group["lr"] for param_group in optimizer.param_groups]


def get_conditions_str(geometry, energy, phi, theta):
    return f"Geo_{geometry}_E_{energy}_Phi_{phi}_Theta_{theta}"


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def sum_flat(tensor):
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])


def unwrap_ddp(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model
