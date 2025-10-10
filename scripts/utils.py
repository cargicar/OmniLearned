from sklearn import metrics
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple

import torch.distributed as dist
from torch.distributed import init_process_group, get_rank
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import numpy as np
import matplotlib.pyplot as plt

def Ehistogram(X1, X2, y, gap, energy, spatial_dim=0, title="Ehistogram Comparison", bin_width=0.05):
    """
    Plots two energy histograms on the same canvas for comparison, 
    aggregating data across all point clouds in the batch.

    Args:
        X1 (torch.Tensor): The first data tensor of shape (batch_size, num_particles, 4).
        X2 (torch.Tensor): The second data tensor of shape (batch_size, num_particles, 4).
        spatial_dim (int): The spatial dimension to use for binning (0 for x, 1 for y, 2 for z).
        title (str): The title for the plot and the filename for saving.
        bin_width (float): The width of each spatial bin.
    """
    # 1. X1 (Dataset) - Aggregate all batches
    # X1[:, :, 3] selects all batch elements, all particles, and the 4th feature (energy).
    # .flatten() combines the batch and particle dimensions.
    energy1 = X1[:, :, 3].detach().cpu().numpy().flatten()
    x_positions1 = X1[:, :, spatial_dim].detach().cpu().numpy().flatten()
    
    # 2. X2 (Generated) - Aggregate all batches
    energy2 = X2[:, :, 3].detach().cpu().numpy().flatten()
    x_positions2 = X2[:, :, spatial_dim].detach().cpu().numpy().flatten()

    # The aggregation loop (for i in range(min(10, X1.shape[0])):) is removed.
    # The subsequent code will now operate on the combined, flattened data.

    # 3. Combine data to determine the global bin edges
    all_x_positions = np.concatenate([x_positions1, x_positions2])
    # Handle the case where all_x_positions might be empty (e.g., if X1/X2 is empty)
    if all_x_positions.size == 0:
        print("Warning: Input data (X1 or X2) is empty. Cannot generate histogram.")
        return
        
    min_x = np.floor(all_x_positions.min() / bin_width) * bin_width
    max_x = np.ceil(all_x_positions.max() / bin_width) * bin_width
    bin_edges = np.arange(min_x, max_x + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Check if we have any bins
    if len(bin_centers) == 0:
        print("Warning: Bin range is invalid. Cannot generate histogram.")
        return

    # 4. Bin and sum energy for X1
    # Use np.histogram for efficient binning and summation
    total_energy1, _ = np.histogram(x_positions1, bins=bin_edges, weights=energy1)

    # 5. Bin and sum energy for X2
    total_energy2, _ = np.histogram(x_positions2, bins=bin_edges, weights=energy2)

    # 6. Plotting
    plt.figure(figsize=(12, 7))

    # Plot X1 data with a smaller offset
    plt.bar(bin_centers - bin_width/4, total_energy1, width=bin_width/2, 
            edgecolor='black', alpha=0.7, label='Dataset')

    # Plot X2 data with a different offset and color
    plt.bar(bin_centers + bin_width/4, total_energy2, width=bin_width/2, 
            edgecolor='black', alpha=0.7, label='Generated', color='red')

    # Extract single values for filename/title from the first batch element
    # as before, assuming they are constant across the batch or we only care about the first one
    category = int(y[0].detach().cpu().numpy())
    gap_id = int(gap[0].detach().cpu().numpy())
    Penergy = energy[0].detach().cpu().numpy()

    plt.title(f'Total Energy vs. Position Bins (Aggregate) - {title}')
    plt.xlabel(f'Position along dimension {spatial_dim}')
    plt.ylabel('Total Energy per bin')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Ensure the 'results' directory exists before saving (if needed, add import os and os.makedirs)
    # import os
    # os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/Ehisto_{title}_pcat_{category}_gcat_{gap_id}_energy_{Penergy:.2f}_aggregate.png")
    plt.close() # Close the figure to free up memory

    
# def Ehistogram(X1, X2, y, gap, energy, spatial_dim=0, title="Ehistogram Comparison", bin_width=0.05):
#     """
#     Plots two energy histograms on the same canvas for comparison.

#     Args:
#         X1 (torch.Tensor): The first data tensor of shape (batch_size, num_particles, 4).
#         X2 (torch.Tensor): The second data tensor of shape (batch_size, num_particles, 4).
#         spatial_dim (int): The spatial dimension to use for binning (0 for x, 1 for y, 2 for z).
#         title (str): The title for the plot and the filename for saving.
#         bin_width (float): The width of each spatial bin.
#     """
#     # 1. X    
#     # energy1 = X1[:, :, 3].detach().cpu().numpy().flatten()
#     # x_positions1 = X1[:, :, spatial_dim].detach().cpu().numpy().flatten()
    
#     # # 2. Generated
#     # energy2 = X2[:, :, 3].detach().cpu().numpy().flatten()
#     # x_positions2 = X2[:, :, spatial_dim].detach().cpu().numpy().flatten()
#     for i in range(min(10, X1.shape[0])):
#         category = int(y[i].detach().cpu().numpy())
#         gap_id = int(gap[i].detach().cpu().numpy())
#         Penergy = energy[i].detach().cpu().numpy()

#         #NOTE just use first point cloud
#         energy1 = X1[i, :, 3].detach().cpu().numpy().flatten()
#         x_positions1 = X1[0, :, spatial_dim].detach().cpu().numpy().flatten()
        
#         # 2. Generated
#         energy2 = X2[i, :, 3].detach().cpu().numpy().flatten()
#         x_positions2 = X2[i, :, spatial_dim].detach().cpu().numpy().flatten()

#         # 3. Combine data to determine the global bin edges
#         all_x_positions = np.concatenate([x_positions1, x_positions2])
#         min_x = np.floor(all_x_positions.min() / bin_width) * bin_width
#         max_x = np.ceil(all_x_positions.max() / bin_width) * bin_width
#         bin_edges = np.arange(min_x, max_x + bin_width, bin_width)
#         bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

#         # 4. Bin and sum energy for X1
#         total_energy1 = np.zeros(len(bin_edges) - 1)
#         bin_indices1 = np.digitize(x_positions1, bin_edges)
#         for i in range(len(x_positions1)):
#             if 0 < bin_indices1[i] <= len(total_energy1):
#                 total_energy1[bin_indices1[i] - 1] += energy1[i]
        
#         # 5. Bin and sum energy for X2
#         total_energy2 = np.zeros(len(bin_edges) - 1)
#         bin_indices2 = np.digitize(x_positions2, bin_edges)
#         for j in range(len(x_positions2)):
#             if 0 < bin_indices2[j] <= len(total_energy2):
#                 total_energy2[bin_indices2[j] - 1] += energy2[j]

#         # 6. Plotting
#         plt.figure(figsize=(12, 7))

#         # Plot X1 data with a smaller offset
#         plt.bar(bin_centers - bin_width/4, total_energy1, width=bin_width/2, 
#                 edgecolor='black', alpha=0.7, label='Dataset')

#         # Plot X2 data with a different offset and color
#         plt.bar(bin_centers + bin_width/4, total_energy2, width=bin_width/2, 
#                 edgecolor='black', alpha=0.7, label='Generated', color='red')

#         plt.title(f'Total Energy vs. Position Bins - {title}')
#         plt.xlabel(f'Position along dimension {spatial_dim}')
#         plt.ylabel('Total Energy per bin')
#         plt.legend()
#         plt.grid(axis='y', linestyle='--', alpha=0.6)
#         plt.savefig(f"results/Ehisto_{title}_pcat_{category}_gcat_{gap_id}_energy_{Penergy:.2f}.png")

def plot_batch_3d(batch_of_point_clouds: torch.Tensor, cates, gaps= None, energies= None, title="pointcloud", exclude_too_small= False):
    """
    Plots each individual point cloud from a batch in a separate 3D scatter plot,
    excluding points where (x, y, z) == (0, 0, 0).

    Args:
        batch_of_point_clouds: A PyTorch tensor of shape (B, N, 3).
        cates, gaps, energies: Tensors containing corresponding metadata for each point cloud.
    """
    
    # Loop through a maximum of 10 point clouds in the batch
    # num_samples = batch_of_point_clouds.shape[0] # To plot the whole batch
    for i in range(min(10, batch_of_point_clouds.shape[0])):
        
        # --- Data Preparation ---
        
        # Extract and convert the current point cloud tensor to a NumPy array
        point_cloud = batch_of_point_clouds[i].detach().cpu().numpy()
        coords = point_cloud[:, :3]
        threshold = 1e-2
        x = point_cloud[:, 0]
        y = point_cloud[:, 1]
        z = point_cloud[:, 2]
        # Extract and convert metadata
        category = int(cates[i].detach().cpu().numpy())
        if gaps is not None:
            gap = int(gaps[i].detach().cpu().numpy())
        if energies is not None:
            energy = energies[i].detach().cpu().numpy()
        # --- Filtering Step: Remove (0, 0, 0) points ---
        # Create a boolean mask: True if ANY coordinate is non-zero
        # np.set_printoptions(threshold=np.inf)
        # xs = x.sort()
        # ys = y.sort()
        # zs = z.sort()
        # print(f"X {x}")
        # print(f"Y {y}")
        # print(f"Z {z}")
        if exclude_too_small:
            # 1. Check which coordinates have an absolute value > threshold
            too_small_mask = np.abs(coords) > threshold    
            # 2. A point is flagged for REMOVAL if ANY of its (x, y, z) coords are too large.
            mask = np.any(too_small_mask, axis=1)
            # --------------------------------------------------------------------
            # Apply the mask to keep only the small-coordinate points
            filtered_coords = coords[mask]
            x = filtered_coords[:, 0]
            y = filtered_coords[:, 1]
            z = filtered_coords[:, 2]

        # Check if the filtered point cloud is empty (highly unlikely but good practice)
        if len(x) == 0:
            print(f"Point Cloud {i+1} is empty after filtering (all points were 0,0,0). Skipping plot.")
            continue
            
        # --- Plotting ---
        
        # Create a new figure and a 3D subplot
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the filtered points
        ax.scatter(x, y, z, s=1)  # s is the marker size

        # Set axis labels and a title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if gaps is not None and energies is not None:
            plot_title = f'Point Cloud {i+1}, {title} particle {category}, gap {gap}, energy {energy:.2f} ({len(x)} pts)'
            plt.savefig(f"results/gen_RF_{i}_{title}_pcat_{category}_gcat_{gap}_energy_{energy:.2f}.png")
        plot_title = f'Point Cloud {i+1}, ({len(x)} pts)'
        plt.savefig(f"results/gen_RF_{i}_{title}_pcat_{category}.png")
        ax.set_title(plot_title)
        
        # Display the plot and save
        
        plt.close()

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
