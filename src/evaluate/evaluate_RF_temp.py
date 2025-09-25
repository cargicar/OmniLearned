""" Naive sampling with RF. Unify this with train_RF.py later. """
import rootutils
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F

from rectified_flow.rectified_flow import RectifiedFlow 
from rectified_flow.samplers.base_sampler import Sampler
from rectified_flow.flow_components.interpolation_solver import AffineInterp
from rectified_flow.utils import visualize_2d_trajectories_plotly, set_seed

rootutils.setup_root(__file__, pythonpath=True)
#from src.models.omnilearnedv2 import PET3
from src.models.omnilearned import PET2
#from dataloader import load_data
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
#from pytorch_optimizer import Lion
#from lion_pytorch import Lion
from src.data.dataset import HDF5Dataset, pad_collate_fn, PklDataset, ShapeNetCore
from src.data.transforms import MinMaxNormalize, CentroidNormalize, Compose

from src.utils import (
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
from src.data.dataset import ShapeNetCore

import argparse
import os # Import os for default path if needed



def parse_arguments():
    """
    Parses command-line arguments for the model training script.

    Returns:
        argparse.Namespace: An object containing all the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Run model training with specified configurations.")
# --- General/Output Arguments ---
     
    # parser.add_argument("--path", type=str, default='/pscratch/sd/c/ccardona/datasets/G4_individual_sims_pkl_test',
    #                      help="Base path to the dataset directory.")
    
    parser.add_argument("--path", type=str, default="/data/ccardona/datasets/G4_individual_sims_pkl_test",
                        help="Base path to the dataset directory.")
    # parser.add_argument("--outdir", type=str, default="/pscratch/sd/c/ccardona/models/G4/",
    #                       help="Output directory for logs, checkpoints, and results.")
    parser.add_argument("--indir", type=str, default="/data/ccardona/models/G4",
                       help="Output directory for logs, checkpoints, and results.")
    parser.add_argument("--save_tag", type=str, default="detector_cats",
                        help="Tag to append to saved files (e.g., model checkpoints, logs).")
    parser.add_argument("--pretrain_tag", type=str, default="pretrain",
                        help="Tag to use when loading pre-trained models.")
    parser.add_argument("--dataset", type=str, default="top",
                        help="Name of the dataset to use (e.g., 'top').")
    parser.add_argument("--wandb", action="store_true", # Use store_true for boolean flags
                        help="Enable Weights & Biases logging.")
   
    
    # RF parameters 
    parser.add_argument("--interp", type=str, default="straight",
        help="Interpolation method for the rectified flow. Choose between ['straight', 'slerp', 'ddim'].",)
    parser.add_argument("--source_distribution", type=str,default="normal",
        help="Distribution of the source samples. Choose between ['normal'].",)
    parser.add_argument("--is_independent_coupling", type=bool, default=True,
        help="Whether training 1-Rectified Flow",)
    parser.add_argument("--train_time_distribution", type=str, default="uniform",
        help="Distribution of the training time samples. Choose between ['uniform', 'lognormal', 'u_shaped'].",)
    parser.add_argument("--train_time_weight", type=str, default="uniform",
        help="Weighting of the training time samples. Choose between ['uniform'].",)
    parser.add_argument("--num_steps", type=int, default=100,
                        help="Number of steps for generation.")

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
    for i in range(batch_size):
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
        plt.savefig(f"results/gen_RF_{i}_{title}_pcat_{category}_gcat_{gap}_energy_{energy}.png")
        plt.close()

def Ehistogram(X1, X2, spatial_dim=0, title="Ehistogram Comparison", bin_width=0.05):
    """
    Plots two energy histograms on the same canvas for comparison.

    Args:
        X1 (torch.Tensor): The first data tensor of shape (batch_size, num_particles, 4).
        X2 (torch.Tensor): The second data tensor of shape (batch_size, num_particles, 4).
        spatial_dim (int): The spatial dimension to use for binning (0 for x, 1 for y, 2 for z).
        title (str): The title for the plot and the filename for saving.
        bin_width (float): The width of each spatial bin.
    """
    # 1. X    
    # energy1 = X1[:, :, 3].detach().cpu().numpy().flatten()
    # x_positions1 = X1[:, :, spatial_dim].detach().cpu().numpy().flatten()
    
    # # 2. Generated
    # energy2 = X2[:, :, 3].detach().cpu().numpy().flatten()
    # x_positions2 = X2[:, :, spatial_dim].detach().cpu().numpy().flatten()

    #NOTE just use first point cloud
    energy1 = X1[0, :, 3].detach().cpu().numpy().flatten()
    x_positions1 = X1[0, :, spatial_dim].detach().cpu().numpy().flatten()
    
    # 2. Generated
    energy2 = X2[0, :, 3].detach().cpu().abs().numpy().flatten() #NOTE abs cheating
    x_positions2 = X2[0, :, spatial_dim].detach().cpu().numpy().flatten()

    # 3. Combine data to determine the global bin edges
    all_x_positions = np.concatenate([x_positions1, x_positions2])
    min_x = np.floor(all_x_positions.min() / bin_width) * bin_width
    max_x = np.ceil(all_x_positions.max() / bin_width) * bin_width
    bin_edges = np.arange(min_x, max_x + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # 4. Bin and sum energy for X1
    total_energy1 = np.zeros(len(bin_edges) - 1)
    bin_indices1 = np.digitize(x_positions1, bin_edges)
    for i in range(len(x_positions1)):
        if 0 < bin_indices1[i] <= len(total_energy1):
            total_energy1[bin_indices1[i] - 1] += energy1[i]
    
    # 5. Bin and sum energy for X2
    total_energy2 = np.zeros(len(bin_edges) - 1)
    bin_indices2 = np.digitize(x_positions2, bin_edges)
    for i in range(len(x_positions2)):
        if 0 < bin_indices2[i] <= len(total_energy2):
            total_energy2[bin_indices2[i] - 1] += energy2[i]

    # 6. Plotting
    plt.figure(figsize=(12, 7))

    # Plot X1 data with a smaller offset
    plt.bar(bin_centers - bin_width/4, total_energy1, width=bin_width/2, 
            edgecolor='black', alpha=0.7, label='Dataset')

    # Plot X2 data with a different offset and color
    plt.bar(bin_centers + bin_width/4, total_energy2, width=bin_width/2, 
            edgecolor='black', alpha=0.7, label='Generated', color='red')

    plt.title(f'Total Energy vs. Position Bins - {title}')
    plt.xlabel(f'Position along dimension {spatial_dim}')
    plt.ylabel('Total Energy per bin')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.savefig(f'results/{title}.png')
    
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

def gen(
    model,
    dataloader,
    device="cuda" if torch.cuda.is_available() else "cpu",
    interp = "straight",
    num_steps= 100,
):

    #FIXME hardcoded
    data_shape = (500,4)
    
    
    straight_rf = RectifiedFlow(
        data_shape= data_shape,#(32, 32),
        velocity_field=model,
        #interp = AffineInterp(name= "logsnr", alpha=alpha_function, beta=sigma_function),
        interp = interp,
        source_distribution="normal",
        # is_independent_coupling=True,
        # train_time_distribution="uniform",
        # train_time_weight="uniform",
        criterion="mse",
        device=device,
    )

  
    model.eval()
    pts = []
    iterdata = iter(dataloader)
    batch = next(iterdata)
    #X, y = batch["X"].to(device, dtype=torch.float), batch["y"].to(device)
    X, energy, y, gap_pid = batch
    X, energy, y, gap_pid = X.to(device), energy.to(device), y.to(device), gap_pid.to(device)
    y = (y == 2).long()
    x_e, x_g, y_e, y_g, e_energ, g_energ = X[y==0], X[y==1], y[y==0], y[y==1], energy[y==0], energy[y==1]

    #plot_batch_3d(x_e, y_e, gap_pid, energy, title = "DATASET")
    model = model.module if hasattr(model, "module") else model
    model_kwargs = {
        key: (batch[key].to(device) if batch[key] is not None else None)
        for key in ["cond", "pid", "add_info"]
        if key in batch
    }
    with torch.no_grad():                 
        euler_sampler = MyEulerSampler(
            rectified_flow=straight_rf,
            num_steps=num_steps,
            num_samples=10,
        )

        # Sample method 1)
        # Will use the default num_steps and num_samples previously set in the Sampler class
        traj1 = euler_sampler.sample_loop(
            seed=233,
            y=y,
            gap= gap_pid,
            energy=energy,
            )
            
        # Sample method 2)
        # We can pass in a custom x_0 to sample from
        set_seed(233)
#        x_0 = straight_rf.sample_source_distribution(batch_size=args.batch)
#        traj2 = euler_sampler.sample_loop(x_0=x_0)

        # Sample method 3)
        # If we pass in num_steps and num_samples, it will override the default values
        #traj3 = euler_sampler.sample_loop(seed=233, num_steps=50, num_samples=10)

        
        #pts = RF_sampler(model, X, y, gap_pid, energy, 10, 500)
        #plot_batch_3d(outputs["x_body"], title = "from model x_body")
        pts= traj1.x_t
        gx_e, gx_g= pts[y==0], pts[y==1]
        Ehistogram(x_e, gx_e, title=f"Ehisto_generated_E_primary_{e_energ[0]}")
        #plot_batch_3d(gx_e, y_e, gap_pid, energy, title = "from model sampler")
        breakpoint()
    
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
        weights_only=False,# Added to avoid issues with optimizer state dict (new in PyTorch 2.0
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
    model = PET2(
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
    #FIXME hardcoded path for dev and deb
    #dataset_path = f"/home/carlos/Rnet_local/datasets/shapenetCore/"
    pkl_files_path = args.path
    dataset = PklDataset(pkl_files_path)
    
    print(f"Successfully loaded val dataset with {len(dataset)} total events.")
        
    # Define the split ratios
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

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

    #TODO Transformed dataset. 
    dataset = PklDataset(pkl_files_path, transform=composed_transform)

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
    gen(model, val_loader, device, interp = args.interp, num_steps= args.num_steps)

    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)