import rootutils
import argparse
import logging
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.utils.checkpoint
import copy
import json
import torchvision

from dataclasses import dataclass, asdict
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
rootutils.setup_root(__file__, pythonpath=True)
from src.data.dataset import HDF5Dataset, PklDataset, ShapeNetCore 
from src.data.transforms import MinMaxNormalize, CentroidNormalize, Compose
from src.models.calopodit import DiT, DiTConfig
from scripts.utils import plot_batch_3d

from diffusers.optimization import get_scheduler

from torchvision import transforms
from tqdm.auto import tqdm

#from rectified_flow.models.dit import DiT, DiTConfig
from rectified_flow.rectified_flow import RectifiedFlow
from rectified_flow.samplers import EulerSampler
from rectified_flow.samplers.base_sampler import Sampler

logger = get_logger(__name__)

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
        "--reflow",
        type=bool,
        default=False,
        help="Whether training ReFlow",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help=(
            "Number of steps for generation. Used in training Reflow and/or evaluation"
        ),
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
        #default = "/global/cfs/cdirs/m3246/hep_ai/G4_individual_sims_pkl_test",
        default="/pscratch/sd/c/ccardona/datasets/shapenetCore",
        help="The root directory where the dataset is stored.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        #default="pscratch/sd/c/ccardona/models/G4",
        default="/pscratch/sd/c/ccardona/models/shapenet",
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
        default=1000,
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
        "--num_classes",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gap_classes",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    
    parser.add_argument(
        "--no_energy_cond",  # <-- Renamed the argument for clarity
        action="store_false",
        dest="energy_cond",  # <-- Tell argparse to save the result to args.energy_cond
        default=True,
        help="Flag to disable conditioning on the energy of each point (sets energy_cond to False).",
    )
    parser.add_argument(
        "--no_train",  # <-- Renamed the argument for clarity
        action="store_false",
        dest="train",  # <-- Tell argparse to save the result to args.energy_cond
        default=True,
        help="Flag to disable conditioning on the energy of each point (sets energy_cond to False).",
    )
    parser.add_argument(
        "--in_features",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
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
        default=10_000,
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
        default="cosine",
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
        default="tensorboard",
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

class EMAModel:
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1 - self.decay
                )
            else:
                self.shadow[name] = param.data.clone()
                print(f"Warning: EMA shadow does not contain parameter {name}")

    def save_pretrained(self, save_directory: str, filename: str = "dit"):
        state_dict_cpu = {k: v.cpu() for k, v in self.shadow.items()}
        output_model_file = os.path.join(save_directory, f"{filename}_ema.pt")
        torch.save(state_dict_cpu, output_model_file)
        print(f"Model weights saved to {output_model_file}")

    def load_pretrained(self, save_directory: str, filename: str = "dit"):
        output_model_file = os.path.join(save_directory, f"{filename}_ema.pt")
        if os.path.exists(output_model_file):
            state_dict = torch.load(output_model_file, map_location="cpu")
            mapped_state_dict = {}
            for name, param in state_dict.items():
                if name in self.model.state_dict():
                    model_param = self.model.state_dict()[name]
                    mapped_state_dict[name] = param.to(
                        device=model_param.device, dtype=model_param.dtype
                    )
                else:
                    print(f"Warning: {name} not found in model's state_dict.")
            self.shadow = mapped_state_dict
            print(
                f"EMA weights loaded from {output_model_file} and mapped to model's device and dtype."
            )
        else:
            print(f"No EMA weights found at {output_model_file}")

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name].data)


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


def main(args):
    #NOTE passing here to read max_particles from the args. Look for a way to do it from dataset
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

    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # 1.1 Prepare models
    logger.info("******  preparing models  ******")

    #TODO clean up this config. Delet unused params and add new useful ones.
    DiT_config = DiTConfig(
        #Point transformer config
        nblocks =  4,
        name= "calopodit",
        num_points = args.max_particles,
        energy_cond = args.energy_cond,
        in_features=args.in_features,
        transformer_features = 128, #512 = hidden_size in current implementation
        #DiT config
        num_classes = args.num_classes,
        gap_classes = args.gap_classes if hasattr(args, 'gap_classes') else None,
        out_channels=args.out_channels,
        hidden_size=128,
        depth=13,
        num_heads=8,
        mlp_ratio=4,
        use_long_skip=True,
        final_conv=False,
    )
    model = DiT(DiT_config)
    model.to(accelerator.device, dtype=weight_dtype)
    if args.train:
        model.train().requires_grad_(True)
    else:
        model.eval().requires_grad_(False)

    # 2. Prepare datasets
    logger.info("***l***  preparing datasets  ******")



    #TODO unify for all datasets,. Better make a class that load the dataset and split it
    #TODO read transform paramaeter from somewhere else and pass it to the creation of dataset
    files_path = args.data_root
    if args.dataset == "G4_pkl":#pkl
        dataset = PklDataset(files_path)
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

        dataset = PklDataset(files_path, transform=composed_transform)
        collate_fn=pad_collate_fn

    elif args.dataset == "G4_h5":#h5
        dataset = HDF5Dataset(files_path)
        collate_fn=None
    elif args.dataset == "ShapeNetCore":#shapenetcore
        # python scripts/train_calopodit.py --dataset ShapeNetCore --num_classes 54 --gap_classes 0 --no_energy_cond --out_channels 3 --in_features 3 --max_particles 3000
        cates = ['Airplane', 'Bag', 'Basket', 'Bathtub']
        train_dataset = ShapeNetCore(files_path, cates, max_num_points= args.max_particles, scale_mode='shape_unit', split='train', transform=None)
        val_dataset = ShapeNetCore(files_path, cates, max_num_points= args.max_particles, scale_mode='shape_unit', split='val', transform=None)
        test_dataset = ShapeNetCore(files_path, cates, max_num_points= args.max_particles, scale_mode='shape_unit', split='test', transform=None)
        collate_fn=None
    
    # Use random_split to create the subsets
    if args.dataset != "ShapeNetCore":
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [num_train, num_val, num_test]
        )
    else:
        dataset = train_dataset
    
    accelerator.print(f"Successfully loaded dataset with {len(dataset)} total events.")
        
    # Define the split ratios
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1


    # Calculate the number of samples for each split
    num_events = len(dataset)
    num_train = int(num_events * train_ratio)
    num_val = int(num_events * val_ratio)
    num_test = num_events - num_train - num_val

    print(f"\nDataset split into:")
    print(f"  Training set: {len(train_dataset)} events")
    print(f"  Validation set: {len(val_dataset)} events")
    print(f"  Test set: {len(test_dataset)} events")

    # Create DataLoaders for each subset
    batch_size = args.train_batch_size
    sample_size = args.sample_batch_size
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)#, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=sample_size, shuffle=False, collate_fn=collate_fn)#, num_workers=args.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)#, num_workers=args.num_workers)

    accelerator.print(f"Train dataset len: {len(train_dataloader)}")
    print(f"learning rate: {args.learning_rate}")
    print("************")
    

    # 3. Prepare optimizers
    model_params_with_lr = {"params": model.parameters(), "lr": args.learning_rate}
    params_to_optimize = [model_params_with_lr]

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # 4. Prepare for training
    logger.info("******  preparing for training  ******")

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(accelerator.unwrap_model(model), DiT):
                    unwrap_model = accelerator.unwrap_model(model)
                    unwrap_model.save_pretrained(output_dir, filename="dit")
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")

                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            model = models.pop()
            print(type(model))

            if isinstance(accelerator.unwrap_model(model), DiT):
                load_model = DiT.from_pretrained(input_dir, filename="dit")
                model.load_state_dict(load_model.state_dict())
            else:
                raise ValueError(f"Wrong model supplied: {type(model)=}.")

            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    data_shape = (args.max_particles,args.in_features)  # (N, 4) 4 for (x,y,z,energy)
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
        device=accelerator.device,
        dtype=weight_dtype,
    )

    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_name = "1rf-dit-cifar"
        accelerator.init_trackers(tracker_name, config=vars(args))

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

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
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,  # Only show the progress bar once on each machine.
    )
    
    if args.train:
        for epoch in range(first_epoch, args.num_train_epochs):
            model.train()

            for step, batch in enumerate(train_dataloader):
                models_to_accumulate = [model]
                with accelerator.accumulate(models_to_accumulate):
                    if args.dataset == "ShapeNetCore":#shapenetcore
                        X, y = batch['X'], batch['y'] # X; {B, N, 3}, energy: {B,}, y: {B,}
                        energy = None
                        gap_pid = None 
                    else:
                        X, energy, y, gap_pid = batch # X; {B, N, 4}, energy: {B,}, y: {B,}, gap_pid: {B,}
                        #X, energy, y, gap_pid = X.to(device), energy.to(device), y.to(device), gap_pid.to(device)
                        #FIXME Using two categories for develpment purposes
                        y = (y == 2).long()
                    x_0 = rectified_flow.sample_source_distribution(X.shape[0])
                    t = rectified_flow.sample_train_time(X.shape[0])
                    t= t.squeeze() #FIXME it seems that rectified_flow adjust time shape to x already during sample_train and also during get_loss, which creates a bug
                    #FIXME. It would improve performance if we dont load dataset during reflow
                    # if args.reflow:
                    #     with torch.no_grad():
                    #         traj1 = euler_sampler.sample_loop(
                    #         seed=233,
                    #         y=y,
                    #         gap= gap_pid,
                    #         energy=energy,
                    #         ).trajectories[-1]
                    #     loss = rectified_flow.get_loss(
                    #         x_0=x_0,
                    #         x_1=traj1,
                    #         y= y,
                    #         gap= gap_pid,
                    #         energy=energy,
                    #         t=t,
                    #     )
                    # else:
                    loss = rectified_flow.get_loss(
                        x_0=x_0,
                        x_1=X,
                        y= y,
                        gap= gap_pid,
                        energy=energy,
                        t=t,
                    )

                    accelerator.backward(loss)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1

                    if args.use_ema:
                        model_ema.update()

                    if accelerator.is_main_process:
                        if global_step % args.checkpointing_steps == 0:
                            if args.checkpoints_total_limit is not None:
                                checkpoints = os.listdir(args.output_dir)
                                checkpoints = [
                                    d for d in checkpoints if d.startswith("checkpoint")
                                ]
                                checkpoints = sorted(
                                    checkpoints, key=lambda x: int(x.split("-")[1])
                                )
                                if len(checkpoints) >= args.checkpoints_total_limit:
                                    num_to_remove = (
                                        len(checkpoints) - args.checkpoints_total_limit + 1
                                    )
                                    removing_checkpoints = checkpoints[0:num_to_remove]
                                    logger.info(
                                        f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                    )
                                    logger.info(
                                        f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                    )
                                    for removing_checkpoint in removing_checkpoints:
                                        removing_checkpoint = os.path.join(
                                            args.output_dir, removing_checkpoint
                                        )
                                        shutil.rmtree(removing_checkpoint)
                            save_path = os.path.join(
                                args.output_dir, f"checkpoint-{global_step}"
                            )

                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                            if args.use_ema:
                                model_ema.save_pretrained(save_path, filename="dit")
                                logger.info(f"Saved EMA model to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}

                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= args.max_train_steps:
                    break

            if accelerator.is_main_process:
                if epoch % args.validation_epochs == 0:
                    pass  # will implement later

        accelerator.end_training()
    else:
        for step, batch in enumerate(val_dataloader):
            models_to_accumulate = [model]
            with accelerator.accumulate(models_to_accumulate):
                if args.dataset == "ShapeNetCore":#shapenetcore
                    X, y = batch['X'], batch['y'] # X; {B, N, 3}, energy: {B,}, y: {B,}
                    energy = None
                    gap_pid = None 
                else:
                    X, energy, y, gap_pid = batch # X; {B, N, 4}, energy: {B,}, y: {B,}, gap_pid: {B,}
                    #X, energy, y, gap_pid = X.to(device), energy.to(device), y.to(device), gap_pid.to(device)
                    #FIXME Using two categories for develpment purposes
                    y = (y == 2).long()

                with torch.no_grad():
                    euler_sampler = MyEulerSampler(
                        rectified_flow=rectified_flow,
                        num_steps=args.num_steps,
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
                    #Ehistogram(X,pts, y, gap_pid, energy, title=f"Ehist_calopodit_del")
                    plot_batch_3d(pts, y, title = "Model sampler")
        


if __name__ == "__main__":
    args = parse_args()
    main(args)
