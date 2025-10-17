import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.pointnet_util import index_points, square_distance
from typing import Tuple

class EdgeConvBlock(nn.Module):
    """
    EdgeConv-based block equivalent to the local feature aggregation in the
    provided Point Transformer block.

    It computes edge features and aggregates them using Max-Pooling.
    """
    def __init__(self, in_features, transformer_features, d_model, k) -> None:
        super().__init__()
        # NOTE: d_model is not strictly necessary but we keep the parameter for consistency.
        # EdgeConv uses the input feature dimension (or a multiple thereof).
        
        self.k = k
        
        # 1. Feature Expansion (Equivalent to self.fc0)
        # We start with a feature projection of the input point coordinates/features.
        self.initial_proj = nn.Sequential(
            nn.Linear(in_features, transformer_features), # e.g., 3 -> 32
            nn.ReLU(),
            nn.Linear(transformer_features, transformer_features) # e.g., 32 -> 32
        )
        
        # 2. EdgeConv Core MLP (Calculates edge feature h_Theta)
        # The input feature for the MLP is the concatenation of:
        # [center_point_feature (32) , displacement_vector (32)] -> Total 64
        # We use a simple 3-layer MLP for feature refinement, outputting 'd_model' features.
        self.edge_mlp = nn.Sequential(
            nn.Conv2d(transformer_features * 2, d_model, kernel_size=1, bias=False), # e.g., 64 -> 512
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=1, bias=False) # Output: d_model (512)
        )
        
        # 3. Final Projection (Equivalent to self.fc2, maps back to original dim)
        self.final_proj = nn.Sequential(
            nn.Linear(d_model, transformer_features), # e.g., 512 -> 32
            nn.ReLU()
        )

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # 0. Initial Feature Projection (Equivalent to self.fc0(xyz) in PT)
        # We use the raw xyz as input, similar to your PT block.
        features = self.initial_proj(xyz) # (b, n, transformer_features=32)
        pre = features # Save for residual connection
        
        B, N, F = features.shape
        
        # 1. k-NN Search
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_features = index_points(features, knn_idx) # b x n x k x 32
        
        # 2. EdgeConv Feature Construction
        # Center feature: x_i -> (b, n, 1, 32)
        center_feature = features[:, :, None, :] 
        
        # Edge feature: x_j - x_i -> (b, n, k, 32)
        # Displacement vector (x_j - x_i): (b, n, k, 32)
        # NOTE: This assumes features for x_j - x_i can be calculated via features_j - features_i
        # which is common in EdgeConv, but usually it's applied to coordinates.
        
        # A more direct EdgeConv formulation using features and displacement:
        # Replicate center feature for k neighbors: (b, n, k, 32)
        center_feature_k = center_feature.expand(B, N, self.k, F)

        # Concatenate center feature and neighbor feature along the feature dimension (F)
        # Input shape for Conv2d needs to be (B, 2*F, N, K)
        edge_input = torch.cat([center_feature_k, knn_features], dim=-1) # (B, N, K, 2*F)
        
        # Permute to (B, C, N, K) for Conv2d
        edge_input = edge_input.permute(0, 3, 1, 2) # (B, 2*F, N, K)
        
        # 3. Apply Edge MLP (h_Theta)
        # edge_mlp output shape: (B, d_model, N, K)
        edge_features = self.edge_mlp(edge_input) 
        
        # 4. Aggregation (Max-Pooling)
        # Max-pool over the K neighbors dimension (axis 3)
        # Aggregated feature shape: (B, d_model, N, 1) -> (B, d_model, N)
        aggregated_features = edge_features.max(dim=-1, keepdim=False)[0] 
        
        # Permute back to (B, N, d_model)
        aggregated_features = aggregated_features.permute(0, 2, 1) # (B, N, d_model)

        # 5. Final Projection and Residual Connection
        # Final Projection: (B, N, d_model) -> (B, N, 32)
        res = self.final_proj(aggregated_features)
        
        # Residual connection
        output_features = res + pre
        
        # We return a dummy tensor for compatibility with calodpodit output structure, so we can plug&play Edge_conv as replacement of PT block
        dummy_attn = torch.zeros((B, N, self.k, self.k), device=output_features.device)
        
        return output_features, dummy_attn