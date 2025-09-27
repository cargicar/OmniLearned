import rootutils
import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np
import torch
import torch.utils.checkpoint
import copy
import json
import torchvision
import matplotlib.pyplot as plt

from dataclasses import dataclass, asdict
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
from diffusers.optimization import get_scheduler
rootutils.setup_root(__file__, pythonpath=True)

from torchvision import transforms
from tqdm.auto import tqdm
from src.data.dataset import HDF5Dataset, PklDataset, ShapeNetCore 
from src.data.transforms import MinMaxNormalize, CentroidNormalize, Compose
from src.models.calopodit import DiT, DiTConfig
from scripts.utils import plot_batch_3d

from rectified_flow.rectified_flow import RectifiedFlow
from rectified_flow.samplers.base_sampler import Sampler



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--interp",
        type=str,
        default="straight",
        help="Interpolation method for the rectified flow. Choose between ['straight', 'slerp', 'ddim'].",
    )
    parser.add_argument(
        "--source_distribution",
        type=str,
        default="normal",
        help="Distribution of the source samples. Choose between ['normal'].",
    )
    parser.add_argument("--dataset", type=str, default="G4_pkl",
                        help="Name of the dataset to use (e.g., 'G4_pkl').")
    parser.add_argument(
        "--is_independent_coupling",
        type=bool,
        default=True,
        help="Whether training 1-Rectified Flow",
    )
    parser.add_argument(
        "--validation",
        type=bool,
        default=True,
        help="Whether compare generation with dataset",
    )
    parser.add_argument(
        "--train_time_distribution",
        type=str,
        default="uniform",
        help="Distribution of the training time samples. Choose between ['uniform', 'lognormal', 'u_shaped'].",
    )
    parser.add_argument(
        "--train_time_weight",
        type=str,
        default="uniform",
        help="Weighting of the training time samples. Choose between ['uniform'].",
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="mse",
        help="Criterion for the rectified flow. Choose between ['mse', 'l1', 'lpips'].",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run dreambooth validation every X epochs. Dreambooth validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`."
        ),
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/ccardona/datasets/G4_individual_sims_pkl_test",
        help="The root directory where the CIFAR-10 dataset is stored.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/ccardona/models/G4",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--use_ema",
        action="store_true",
        help="Whether to use an exponential moving average of the model weights.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=32,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--max_particles",
        type=int,
        default=1024,
        help=(
            "The maximun number of particles in each point cloud"
        ),
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=64,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--sample_batch_size",
        type=int,
        default=4,
        help="Batch size (per device) for sampling images.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=500_000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=20_000,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
            " checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default="latest",
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument(
        "--lr_power",
        type=float,
        default=1.0,
        help="Power factor of the polynomial scheduler.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=4,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam and Prodigy optimizers.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-04,
        help="Weight decay to use for unet params",
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    args = parser.parse_args()

    return args

class MyEulerSampler(Sampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def step(self, **model_kwargs):
        # Extract the current time, next time point, and current state
        t, t_next, x_t = self.t, self.t_next, self.x_t
        # Compute the velocity field at the current state and time
        v_t = self.rectified_flow.get_velocity(x_t=x_t, t=t, **model_kwargs)
        
        # Update the state using the Euler formula
        self.x_t = x_t + (t_next - t) * v_t
     
    def record(self):
        """
        Overrides the base class method to prevent recording trajectories.
        """
        pass

def plot_image_batch(images):
    """
    Plots a batch of images from a PyTorch tensor.
    
    Args:
        images (torch.Tensor): A tensor of shape [B, C, H, W]
    """
    # Create a figure and a set of subplots
    fig, axes = plt.subplots(1, images.shape[0], figsize=(10, 3))
    
    # Check if there's only one image in the batch and adjust axes
    if images.shape[0] == 1:
        axes = [axes]
        
    for i, ax in enumerate(axes):
        # 1. Denormalize image from [-1, 1] to [0, 1]
        img = images[i] / 2 + 0.5
        
        # 2. Permute the dimensions from [C, H, W] to [H, W, C]
        img = img.permute(1, 2, 0)
        
        # 3. Convert to a NumPy array for plotting
        np_img = img.detach().cpu().numpy()
        
        # Plot the image
        ax.imshow(np_img)
        ax.axis('off')  # Hide axes
        ax.set_title(f'Sample {i+1}')
    
    plt.tight_layout()
    plt.savefig(f"results/cifar_batch_{i}")

def main(args):
    #FIXME and passing here to read max_particles from the args. Look for a way to do it from dataset
    def pad_collate_fn(batch, max_particles=args.max_particles):
        """
        Custom collate function to handle batches of showers with varying numbers of particles.
        It pads or truncates each shower to a fixed size and then stacks them.

        Args:
            batch (list): A list of data samples from the dataset.
            max_particles (int): The maximum number of particles to keep per shower.
        Returns:
            A tuple of batched PyTorch tensors.
        """
        showers_list, energies_list, pids_list, gap_pids_list = zip(*batch)

        padded_showers = []
        for shower in showers_list:
            num_particles = shower.shape[0]
            if num_particles > max_particles:
                # Truncate if the number of particles exceeds the max
                padded_shower = shower[:max_particles]
            else:
                # Pad with zeros if the number of particles is less than the max
                padding = torch.zeros(max_particles - num_particles, shower.shape[1], dtype=shower.dtype, device=shower.device)
                padded_shower = torch.cat([shower, padding], dim=0)
            padded_showers.append(padded_shower)

        # Stack all tensors to create the batch
        showers_batch = torch.stack(padded_showers, dim=0)
        energies_batch = torch.stack(energies_list, dim=0)
        pids_batch = torch.stack(pids_list, dim=0)
        gap_pids_batch = torch.stack(gap_pids_list, dim=0)

        return showers_batch, energies_batch, pids_batch, gap_pids_batch

    #TODO add accelerator
    device="cuda" if torch.cuda.is_available() else "cpu"    
   #TODO clean up this config. Delet unused params and add new useful ones.
    DiT_config = DiTConfig(
        #Point transformer config
        nblocks =  4,
        name= "calopodit",
        num_points = args.max_particles,
        #num_centroids = 128,
        in_features=4,
        transformer_features = 128, #512 = hidden_size in current implementation
        #DiT config
        num_classes = 2,
        gap_classes = 4,
        out_channels=4,
        hidden_size=128,
        depth=13,
        num_heads=8,
        mlp_ratio=4,
        use_long_skip=True,
        final_conv=False,
    )
    model = DiT(DiT_config)
    weight_dtype = torch.float32
    model.to(device, dtype=weight_dtype)

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None
        if path is None:
            print(f"Checkpoint '{args.resume_from_checkpoint}' does not exist.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            print(f"Using checkpoint {path}")
            checkpoint = torch.load(os.path.join(args.output_dir, path, "dit_model.pt"),
                                    map_location=device,
                                    weights_only=False
                                    )
            model.load_state_dict(checkpoint)        

    print(f"Resuming from checkpoint {path}")
    model.eval().requires_grad_(False)

    if args.validation: 
        
        #TODO unify for all datasets,. Better make a class that load the dataset and split it
        #TODO read transform paramaeter from somewhere else and pass it to the creation of dataset
        files_path = args.data_root
        if args.dataset == "G4_pkl":#pkl
            dataset = PklDataset(files_path)
        elif args.dataset == "G4_h5":#h5
            dataset = HDF5Dataset(files_path)
        elif args.dataset == "ShapeNetCore":#shapenetcore
            dataset = ShapeNetCore(files_path)
        

        #Transforms
        #TODO read from detector geometries (?)
        min_vals = dataset.all_showers.min(axis=(0,1))
        max_vals = dataset.all_showers.max(axis=(0,1))
        # 4. Create the normalization transform object with these values
        
        centroid_transform = CentroidNormalize()
        minmax_transform = MinMaxNormalize(min_vals, max_vals)

        composed_transform = Compose([
                            centroid_transform,
                            minmax_transform
                            ])
    
        # Transformed dataset.
        if args.dataset == "G4_pkl":#pkl # Change name to anything if you wnat to unnormalized quickly
            dataset = PklDataset(files_path, transform=composed_transform)
        elif args.dataset == "G4_h5":#h5
            dataset = HDF5Dataset(files_path) #TODO, transform=normalize_transform)
        elif args.dataset == "ShapeNetCore":#shapenetcore
            dataset = ShapeNetCore(files_path)#TODO, transform=normalize_transform)
        
        
                
        # Define the split ratios
        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1


        # Calculate the number of samples for each split
        num_events = len(dataset)
        num_train = int(num_events * train_ratio)
        num_val = int(num_events * val_ratio)
        num_test = num_events - num_train - num_val

        # Use random_split to create the subsets
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [num_train, num_val, num_test]
        )

        print(f"\nDataset split into:")
        print(f"  Training set: {len(train_dataset)} events")
        print(f"  Validation set: {len(val_dataset)} events")
        print(f"  Test set: {len(test_dataset)} events")

        # Create DataLoaders for each subset
        batch_size = args.train_batch_size
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)#, num_workers=args.num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)#, num_workers=args.num_workers)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)#, num_workers=args.num_workers)


        print(f"Train dataset len: {len(train_dataloader)}")
        print("************")

        batch_size= args.sample_batch_size

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )

        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )

        # You can now use train_loader for training and val_loader for validation

        iterdata = iter(val_loader)
        batch = next(iterdata)
        X, energy, y, gap_pid = batch
        X, energy, y, gap_pid = X.to(device), energy.to(device), y.to(device), gap_pid.to(device)
    
    #FIXME features hardcoded
    data_shape = (args.max_particles, 4)
    # wrap up rectified_flow after accelerator.prepare
    rectified_flow = RectifiedFlow(
        data_shape=data_shape,
        interp=args.interp,
        source_distribution=args.source_distribution,
        is_independent_coupling=args.is_independent_coupling,
        train_time_distribution=args.train_time_distribution,
        train_time_weight=args.train_time_weight,
        criterion=args.criterion,
        velocity_field=model,
        device=device,
        dtype=weight_dtype,
    )
    with torch.no_grad():                 
        euler_sampler = MyEulerSampler(
            rectified_flow=rectified_flow,
            num_steps=100, #FIXME Add parser flag
            num_samples=args.sample_batch_size,
        )

        # Sample method 1)
        # Will use the default num_steps and num_samples previously set in the Sampler class
        traj1 = euler_sampler.sample_loop(
            seed=233,
            y=y,
            gap= gap_pid,
            energy=energy,
            )
        pts= traj1.x_t
        plot_batch_3d(pts, y, gap_pid, energy, title = "model sampler")
        
if __name__ == "__main__":
    args = parse_args()
    main(args)