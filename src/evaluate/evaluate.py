import rootutils
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
rootutils.setup_root(__file__, pythonpath=True)
from src.models.omnilearned import PET2
from src.models.calolearned import PET3
#from dataloader import load_data
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
#from pytorch_optimizer import Lion
#from lion_pytorch import Lion
from diffusers.optimization import get_cosine_schedule_with_warmup
from src.data.dataset import HDF5Dataset, pad_collate_fn, PklDataset, ShapeNetCore

from scripts.utils import (
    is_master_node,
    ddp_setup,
    get_param_groups,
    CLIPLoss,
    get_checkpoint_name,
)
import time
import os
import torch.amp as amp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import argparse
import os # Import os for default path if needed
#from diffusion import sampler


def parse_arguments():
    """
    Parses command-line arguments for the model training script.

    Returns:
        argparse.Namespace: An object containing all the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run model training with specified configurations.")
# --- General/Output Arguments ---
    
    parser.add_argument("--path", type=str, default='/pscratch/sd/c/ccardona/datasets/G4_individual_sims_pkl_test',
                         help="Base path to the dataset directory.")
    
    #parser.add_argument("--path", type=str, default="/pscratch/sd/c/ccardona/datasets/G4_h5/all_sims_combined.h5",
    #                     help="Base path to the dataset directory.")
    
    parser.add_argument("--indir", type=str, default="/pscratch/sd/c/ccardona/models/G4/",
                          help="Output directory for logs, checkpoints, and results.")
    #parser.add_argument("--outdir", type=str, default="/home/carlos/Rnet_local/saved_models",
    #                    help="Output directory for logs, checkpoints, and results.")
    parser.add_argument("--save_tag", type=str, default="detector_cats",
                        help="Tag to append to saved files (e.g., model checkpoints, logs).")
    parser.add_argument("--pretrain_tag", type=str, default="pretrain",
                        help="Tag to use when loading pre-trained models.")
    parser.add_argument("--dataset", type=str, default="top",
                        help="Name of the dataset to use (e.g., 'top').")
    parser.add_argument("--wandb", action="store_true", # Use store_true for boolean flags
                        help="Enable Weights & Biases logging.")
   
    # --- Training State Arguments ---
    parser.add_argument("--fine_tune", action="store_true",
                        help="Enable fine-tuning mode (loads pre-trained weights and adjusts learning rate).")
    parser.add_argument("--resuming", action="store_true",
                        help="Resume training from the latest checkpoint in outdir/save_tag.")

    # --- Data/Feature Arguments ---

    parser.add_argument("--num_feat", type=int, default=4,
                        help="Number of features per particle/vector (e.g., 4 for 4-vectors).")
    parser.add_argument("--conditional", action="store_true",
                        help="Enable conditional generation/training.")
    
    parser.add_argument("--use_interaction", action="store_true",
                        help="Enable interaction block.")
    parser.add_argument("--num_cond", type=int, default=3,
                        help="Number of conditioning features/dimensions.")
    parser.add_argument("--use_pid", action="store_true",
                        help="Use Particle ID (PID) as an input feature.")
    parser.add_argument("--pid_idx", type=int, default=-1,
                        help="Index of the PID feature in the input data (if use_pid is True).")
    parser.add_argument("--use_add", action="store_true",
                        help="Use additional features.")
    parser.add_argument("--num_add", type=int, default=4,
                        help="Number of additional features.")
    parser.add_argument("--use_clip", action="store_true",
                        help="Enable gradient clipping.")
    parser.add_argument("--use_event_loss", action="store_true",
                        help="Enable event-level loss calculation.")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="Number of output classes for classification tasks.")
    parser.add_argument("--mode", type=str, default="generator",
                        choices=["classifier", "generator", "other_mode_if_any"], # Add valid choices
                        help="Operating mode of the model (e.g., 'classifier', 'generator').")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of worker processes for data loading.")


    # --- Training Hyperparameters ---
    parser.add_argument("--batch", type=int, default=64,
                        help="Batch size for training.")
    parser.add_argument("--iterations", type=int, default=-1,
                        help="Number of training iterations. If -1, run for specified epochs.")
    parser.add_argument("--epoch", type=int, default=1000,
                        help="Number of training epochs.")
    parser.add_argument("--warmup_epoch", type=int, default=1,
                        help="Number of warmup epochs for learning rate scheduling.")
    parser.add_argument("--use_amp", action="store_true",
                        help="Enable Automatic Mixed Precision (AMP) training.")
    parser.add_argument("--optim", type=str, default="adamw",
                        choices=["adamw", "lion", "sgd"], # Example: add common optimizers
                        help="Optimizer to use (e.g., 'lion', 'adamw').")
    parser.add_argument("--b1", type=float, default=0.95,
                        help="Beta1 parameter for Adam-like optimizers.")
    parser.add_argument("--b2", type=float, default=0.98,
                        help="Beta2 parameter for Adam-like optimizers.")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Initial learning rate.")
    parser.add_argument("--lr_factor", type=float, default=10.0,
                        help="Learning rate factor for fine-tuning or scheduling.")
    parser.add_argument("--wd", type=float, default=0.3,
                        help="Weight decay (L2 regularization).")

    # --- Model Architecture Hyperparameters (if applicable, e.g., for a Transformer) ---
    parser.add_argument("--num_transf", type=int, default=6,
                        help="Number of transformer blocks/layers.")
    parser.add_argument("--num_transf_heads", type=int, default=2,
                        help="Number of attention heads in each transformer block.")
    parser.add_argument("--num_tokens", type=int, default=4,
                        help="Number of tokens in the model (e.g., for certain attention mechanisms).")
    parser.add_argument("--num_head", type=int, default=8,
                        help="General number of attention heads (if different from num_transf_heads).")
    parser.add_argument("--K", type=int, default=15,
                        help="K parameter for K-Nearest Neighbors or similar (e.g., for graph construction).")
    parser.add_argument("--base_dim", type=int, default=64,
                        help="Base dimension for model embeddings/features.")
    parser.add_argument("--mlp_ratio", type=int, default=2,
                        help="MLP hidden dimension ratio relative to base_dim.")
    parser.add_argument("--attn_drop", type=float, default=0.1,
                        help="Dropout rate for attention layers.")
    parser.add_argument("--mlp_drop", type=float, default=0.1,
                        help="Dropout rate for MLP layers.")
    parser.add_argument("--feature_drop", type=float, default=0.0,
                        help="Dropout rate for input features.")


    args = parser.parse_args()
    return args

def plot_batch_3d(batch_of_point_clouds: torch.Tensor, cates, gaps, energies, title="pointcloud"):
    """
    Plots each individual point cloud from a batch in a separate 3D scatter plot.

    Args:
        batch_of_point_clouds: A PyTorch tensor of shape (B, N, 3), where:
            - B is the batch size (e.g., 128)
            - N is the number of points (e.g., 2048)
            - 3 represents the (x, y, z) coordinates
    """
    # Get the batch size
    batch_size = batch_of_point_clouds.shape[0]
    # Loop through each point cloud in the batch
    for i in range(10):
    #for i in range(num_samples):
        # Extract the current point cloud tensor
        # .detach() is used to remove it from the computation graph.
        # .cpu() ensures the tensor is on the CPU.
        # .numpy() converts the tensor to a NumPy array, which matplotlib requires.
        point_cloud = batch_of_point_clouds[i].detach().cpu().numpy()
        category = int(cates[i].detach().cpu().numpy())
        gap = int(gaps[i].detach().cpu().numpy())
        energy = energies[i].detach().cpu().numpy()
        # Separate the coordinates for plotting
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]

        # Create a new figure and a 3D subplot for the current point cloud
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the points
        ax.scatter(x, y, z, s=1)  # s is the marker size

        # Set axis labels and a title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Point Cloud {i+1}, {title} particle {category}, gap {gap}, energy {energy}')
        
        # Display the plot
        plt.savefig(f"results/gen_{i}_{title}_pcat_{category}_gcat_{gap}_energy_{energy}.png")
        plt.close()

def gather_tensors(x):
    """
    If running under DDP, all_gather x from every rank, concat, then return as numpy.
    Otherwise just .cpu().numpy().
    """
    if dist.is_initialized():
        ws = dist.get_world_size()
        # pre‐allocate one buffer per rank
        buf = [torch.zeros_like(x) for _ in range(ws)]
        dist.all_gather(buf, x)
        x = torch.cat(buf, dim=0)
    return x.cpu()


def eval_model(
    model,
    val_loader,
    device="cpu",
):
    start = time.time()
    prediction, mass, pt, labels = test_step(model, val_loader, device)

    if dist.is_initialized():
        prediction, mass, pt, labels = [
            gather_tensors(t) for t in (prediction, mass, pt, labels)
        ]

    if is_master_node():
        print_metrics(prediction.softmax(-1).numpy(), labels.numpy())
    if is_master_node():
        print("Time taken for evaluation is {} sec".format(time.time() - start))
    #     if use_event_loss:
    #         np.savez(f"/pscratch/sd/v/vmikuni/outputs_{dataset}.npz",
    #                  # prediction= torch.nn.functional.log_softmax(prediction[:,:200],dim=-1).numpy(),
    #                  # event_prediction = torch.nn.functional.log_softmax(prediction[:,200:],dim=-1).numpy(),
    #                  prediction= prediction[:,:200].softmax(-1).numpy(),
    #                  event_prediction = prediction[:,200:].softmax(-1).numpy(),
    #                  mass=mass.numpy(), pt=pt.numpy())
    #     else:
    #         np.savez(f"/pscratch/sd/v/vmikuni/outputs_{dataset}.npz",
    #                  prediction= torch.nn.functional.log_softmax(prediction,dim=-1).numpy(),
    #                  mass=mass.numpy(), pt=pt.numpy())


def test_step(
    model,
    dataloader,
    device,
):
    model.eval()

    preds = []
    labels = []
    masses = []
    pts = []

    for ib, batch in enumerate(dataloader):
        if ib > 30000:
            break
        #X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
        X = batch["pointcloud"].to(device, dtype=torch.float)
        y = batch["cate"].to(device)
        plot_batch_3d(X, y, title = "from dataset")
        model_kwargs = {
            key: (batch[key].to(device) if batch[key] is not None else None)
            for key in ["cond", "pid", "add_info"]
            if key in batch
        }
        with torch.no_grad():
            outputs = model(X, y, **model_kwargs)
            #plot_batch_3d(outputs["x_body"], title = "from model x_body")
            plot_batch_3d(outputs["z_body"], y, title = "from model sampler")
        preds.append(outputs["y_pred"])
        labels.append(y)
        masses.append(torch.exp(batch["cond"][:, 1]))
        pts.append(torch.exp(batch["cond"][:, 0]))

    return (
        torch.cat(preds).to(device),
        torch.cat(masses).to(device),
        torch.cat(pts).to(device),
        torch.cat(labels).to(device),
    )


def gen(
    model,
    dataloader,
    device,
):
    model.eval()
    pts = []
    iterdata = iter(dataloader)
    batch = next(iterdata)
    #X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
    X, energy, y, gap_pid = batch
    X, energy, y, gap_pid = X.to(device), energy.to(device), y.to(device), gap_pid.to(device)
    y = (y == 2).long()
    plot_batch_3d(X, y, gap_pid, energy, title = "from dataset")
    model_kwargs = {
        key: (batch[key].to(device) if batch[key] is not None else None)
        for key in ["cond", "pid", "add_info"]
        if key in batch
    }
    x_shape = X.shape
    with torch.no_grad():
        #pts = sampler(model, X, y, gap_pid, energy, 1000, 500, model_kwargs)
        #generated_events = model.sample(conditions=conditions, progress=True, **cfg.sampling).squeeze(1)  # squeeze to remove the output channel dimension
        try:
            z_body = model.body(x_shape, cond=None, pid=None, add_info=None)
            generated_events = model.diffusion.sample(z_body, y, gap_pid, energy, progress=True).squeeze(1)  # squeeze to remove the output channel dimension
        except AttributeError:
            z_body = model.module.body(x_shape, cond=None, pid=None, add_info=None)
            generated_events = model.module.diffusion.sample(z_body, y, gap_pid, energy, progress=True).squeeze(1)  # squeeze to remove the output channel dimension
            
        #plot_batch_3d(outputs["x_body"], title = "from model x_body")
        #plot_batch_3d(pts, y, gap_pid, energy, title = "from model sampler")
        plot_batch_3d(generated_events, y, gap_pid, energy, title = "from model sampler")
    
    # return (
    #     torch.cat(pts).to(device),
    # )



def restore_checkpoint(
    model,
    checkpoint_dir,
    checkpoint_name,
    device,
    is_main_node=False,
):
    device = "cuda:{}".format(device) if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, checkpoint_name),
        map_location=device,
    )

    base_model = model.module if hasattr(model, "module") else model
    base_model.to(device)
    base_model.body.load_state_dict(checkpoint["body"], strict=False)

    if base_model.classifier is not None and "classifier_head" in checkpoint:
        base_model.classifier.load_state_dict(checkpoint["classifier_head"])

    if base_model.generator is not None:
        base_model.generator.load_state_dict(checkpoint["generator_head"])


def main(args):
    local_rank, rank, size = ddp_setup()
    # set up model
    model = PET3(
        input_dim=args.num_feat,
        hidden_size=args.base_dim,
        num_transformers=args.num_transf,
        num_transformers_head=args.num_transf_heads,
        num_heads=args.num_head,
        mlp_ratio=args.mlp_ratio,
        mlp_drop=args.mlp_drop,
        attn_drop=args.attn_drop,
        feature_drop=args.feature_drop,
        num_tokens=args.num_tokens,
        K=args.K,
        conditional=args.conditional,
        cond_dim=args.num_cond,
        pid=args.use_pid,
        add_info=args.use_add,
        add_dim=args.num_add,
        use_time=False if args.mode == "classifier" else True,
        mode=args.mode,
        num_classes=args.num_classes,
    )
    if rank == 0:
        d = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("**** Setup ****")
        print(
            "Total params: %.2fM"
            % (sum(p.numel() for p in model.parameters()) / 1000000.0)
        )
        print(f"Evaluating on device: {d}, with {size} GPUs")
        print("************")

    # load in train data
    # val_loader = load_data(
    #     dataset,
    #     dataset_type="test",
    #     use_pid=use_pid,
    #     pid_idx=pid_idx,
    #     use_add=use_add,
    #     num_add=num_add,
    #     path=path,
    #     batch=batch,
    #     num_workers=num_workers,
    #     rank=rank,
    #     size=size,
    # )
    
    pkl_files_path = args.path
    dataset = PklDataset(pkl_files_path)
    
    print(f"Successfully loaded val dataset with {len(dataset)} total events.")
        
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
    print(f"  Validation set: {len(val_dataset)} events")


    # Create DataLoaders for each subset
    batch_size = args.batch
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=args.num_workers)

    if rank == 0:
        print("**** Setup ****")
        print(f"Train dataset len: {len(val_loader)}")
        print("************")

    if os.path.isfile(os.path.join(args.indir, get_checkpoint_name(args.save_tag))):
        if is_master_node():
            print(
                f"Loading checkpoint from {os.path.join(args.indir, get_checkpoint_name(args.save_tag))}"
            )

        restore_checkpoint(
            model,
            args.indir,
            get_checkpoint_name(args.save_tag),
            local_rank,
        )

    else:
        raise ValueError(
            f"Error loading checkpoint: {os.path.join(args.indir, get_checkpoint_name(args.save_tag))}"
        )

    # Transfer model to GPU if available
    kwarg = {}
    if torch.cuda.is_available():
        device = local_rank
        model.to(local_rank)
        kwarg["device_ids"] = [device]
    else:
        model.cpu()
        device = "cpu"

    model = DDP(
        model,
        **kwarg,
    )

    #eval_model(model, val_loader, device=device)
    gen(model, val_loader, device)

    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)