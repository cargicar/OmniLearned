from src.models.pointnet_util import index_points, square_distance
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransformerBlock(nn.Module):
    def __init__(self, in_features, transformer_features, d_model, k) -> None: #transformer_features: 32
        super().__init__()
        self.fc0 = nn.Sequential(
            nn.Linear(in_features, transformer_features),
            nn.ReLU(),
            nn.Linear(transformer_features, transformer_features)
        )
        self.fc1 = nn.Linear(transformer_features, d_model)
        self.fc2 = nn.Linear(d_model, transformer_features)
        self.fc_delta = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k
        
    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz):
        features = self.fc0(xyz) # (b, n, 32)
        # xyz: bxnx3, features: bxnx32 = xyz*linear(3,32)
        #dist: bxnxn = bx(n_{i,j}= square distance n_i to n_j) (why not square-root?)
        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k = bx(firs_k(distance n_i to k n_j ))
        knn_xyz = index_points(xyz, knn_idx) # b x n x k x 3 = bx(3 dim coordinates of k closest points to n_i)
        # in simplest words, each point n_i has attached to it is closest k neightbors
        pre = features # (xyz: bxnxd * linear(d,32)=bxnx32)
        #projection to attention dim: bxnx32*linear(32xd_model:512)=bxnxd_model
        x = self.fc1(features) 
        # q : (x:bxnx512)*(linear(d_model:d_model,d_model)) = bxnxd_model
        # k : index_points(w_ks(x):bxnx512, knn_idx) = bxnx(k indexes of closes k points to poin n_i)xd_model
        # v isomorphic k
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        #self.dc_delta iso linear(3,d_model:512)
        # xyzk :=(xyz[:,:,None]:bxnx1x3-knn_xyz):bxnxkx3= bxnxkx3 (Per each point position, substracts the position of k closer points (?))
        #self.dc_delta(xyzk): bxnxkx3* linear(3,512)=bxnxkxd_model. Projection of xyzk into attention dim
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x d_model
        # pos_enc is pretty much the projection of knn_xyz into attention dim. 
        # self.fc_gamma iso linear( d_model, d_model)
        # q[:, :, None]:bxnx1xk - k:bxnx[k]xd_model + pos_enc:b x n x k x d_model = b x n x k x d_model
        # attn = bxnxkxd_model*linear(d_model, d_model): bxnxkxd_model
        attn = self.fc_gamma(q[:, :, None] - k + pos_enc)
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x f
        # einstein sumation over k dim: bxnxkxd_model*bxnxkxd_model-> bxnxd_model
        # res attn*(v+pos_enc)
        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        # res:bxnxd_model*linear(d_model,d_points) = bxnx(d_points=32)
        res = self.fc2(res) + pre
        return res, attn
    