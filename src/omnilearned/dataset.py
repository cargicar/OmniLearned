import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from copy import copy

# The provided dictionary mapping synset IDs to category names
synsetid_to_cate = {'02691156': 'Airplane', '02773838': 'Bag', '02801938': 'Basket', '02808440': 'Bathtub', '02818832': 'Bed', '02828884': 'Bench', '02876657': 'Bottle', '02880940': 'Bowl', 
                    '02924116': 'Bus', '02933112': 'Cabinet', '02747177': 'Can', '02942699': 'Camera', '02954340': 'Cap', '02958343': 'Car', '03001627': 'Chair', '03046257': 'Clock', 
                    '03207941': 'Dishwasher', '03211117': 'Monitor', '04379243': 'Table', '04401088': 'Telephone', '02946921': 'Tin_can', '04460130': 'Tower', '04468005': 'Train',
                    '03085013': 'Keyboard', '03261776': 'Earphone', '03325088': 'Faucet', '03337140': 'File', '03467517': 'Guitar', '03513137': 'Helmet', '03593516': 'Jar', '03624134': 'Knife', 
                    '03636649': 'Lamp', '03642806': 'Laptop', '03691459': 'Speaker', '03710193': 'Mailbox', '03759954': 'Microphone', '03761084': 'Microwave', '03790512': 'Motorcycle', '03797390': 'Mug',
                    '03928116': 'Piano', '03938244': 'Pillow', '03948459': 'Pistol', '03991062': 'Pot', '04004475': 'Printer', '04074963': 'Remote_control', '04090263': 'Rifle', '04099429': 'Rocket', 
                    '04225987': 'Skateboard', '04256520': 'Sofa', '04330267': 'Stove', '04530566': 'Vessel', '04554684': 'Washer', '02992529': 'Cellphone', '02843684': 'Birdhouse', '02871439': 'Bookshelf'}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}

class ShapeNetCore(Dataset):

    def __init__(self, path, cates, split, scale_mode, transform=None):
        super().__init__()
        assert isinstance(cates, list), '`cates` must be a list of cate names.'
        assert split in ('train', 'val', 'test')
        assert scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34', None)
        
        self.path = path  # The root directory of the dataset
        self.split = split
        self.scale_mode = scale_mode
        self.transform = transform
        self.stats = None
        
        # Load the split JSON file
        split_path = os.path.join(self.path, f'{self.split}_split.json')
        with open(split_path, 'r') as f:
            data_list = json.load(f)
        # Filter data based on specified categories
        self.data_list = []
        if 'all' in cates:
            self.data_list = data_list
        else:
            cate_set = set(cates)
            for item in data_list:
                _, cate_name, _ = item
                if cate_name in cate_set:
                    self.data_list.append(item)
        
        print(f'Loaded {len(self.data_list)} models for split "{self.split}".')

        # Load pre-computed statistics for global normalization
        if self.scale_mode == 'global_unit':
            stats_path = os.path.join(self.path, '_stats/stats_all.pt')
            if not os.path.exists(stats_path):
                raise FileNotFoundError(f"Global statistics file not found at: {stats_path}. Please pre-compute it.")
            self.stats = torch.load(stats_path)
            print("Loaded global statistics.")


    def __len__(self):
        return len(self.data_list)

    # def __getitem__(self, idx):
    #     # Retrieve info from the filtered list
    #     _, cate_name, file_path_rel = self.data_list[idx]
    #     file_path = os.path.join(self.path, file_path_rel)
        
    #     # Load the point cloud from the .npy file
    #     pc = torch.from_numpy(np.load(file_path))
        
    #     # Apply scaling based on the chosen mode
    #     if self.scale_mode == 'global_unit':
    #         shift = self.stats['mean'].reshape(1, 3)
    #         scale = self.stats['std'].reshape(1, 1)
    #     elif self.scale_mode == 'shape_unit':
    #         shift = pc.mean(dim=0).reshape(1, 3)
    #         scale = pc.flatten().std().reshape(1, 1)
    #     elif self.scale_mode == 'shape_half':
    #         shift = pc.mean(dim=0).reshape(1, 3)
    #         scale = pc.flatten().std().reshape(1, 1) / (0.5)
    #     elif self.scale_mode == 'shape_bbox':
    #         pc_max, _ = pc.max(dim=0, keepdim=True)
    #         pc_min, _ = pc.min(dim=0, keepdim=True)
    #         shift = ((pc_min + pc_max) / 2).view(1, 3)
    #         scale = (pc_max - pc_min).max().reshape(1, 1) / 2
    #     else: # No scaling
    #         shift = torch.zeros([1, 3])
    #         scale = torch.ones([1, 1])

    #     # Apply the transformation
    #     pc = (pc - shift) / scale

    #     data = {
    #         'pointcloud': pc,
    #         'cate': cate_name,
    #         'id': idx, # Use index as a unique ID for this split
    #         'shift': shift,
    #         'scale': scale
    #     }

    #     # Apply any optional external transforms
    #     if self.transform is not None:
    #         data = self.transform(data)

    #     return data
    
    #### New Getitem with max num of poitns
    #FIXME add flag for max num of points
    def __getitem__(self, idx):
        # Retrieve info from the filtered list
        _, cate_name, file_path_rel = self.data_list[idx]
        file_path = os.path.join(self.path, file_path_rel)
        
        # Load the point cloud from the .npy file
        pc = torch.from_numpy(np.load(file_path))
        
        #FIXME
        # Define a fixed number of points for all point clouds
        num_points = 2048  # You can adjust this value

        # Sample or pad the point cloud to the fixed size
        if pc.shape[0] > num_points:
            # Randomly sample 'num_points' from the point cloud
            indices = np.random.choice(pc.shape[0], num_points, replace=False)
            pc = pc[indices]
        elif pc.shape[0] < num_points:
            # Pad with zeros or duplicate points if the point cloud is too small
            # This is a basic padding approach; more advanced methods exist.
            zeros_to_add = torch.zeros(num_points - pc.shape[0], 3)
            pc = torch.cat([pc, zeros_to_add], dim=0)

        # Apply scaling based on the chosen mode
        if self.scale_mode == 'global_unit':
            shift = self.stats['mean'].reshape(1, 3)
            scale = self.stats['std'].reshape(1, 1)
        elif self.scale_mode == 'shape_unit':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1)
        elif self.scale_mode == 'shape_half':
            shift = pc.mean(dim=0).reshape(1, 3)
            scale = pc.flatten().std().reshape(1, 1) / (0.5)
        elif self.scale_mode == 'shape_bbox':
            pc_max, _ = pc.max(dim=0, keepdim=True)
            pc_min, _ = pc.min(dim=0, keepdim=True)
            shift = ((pc_min + pc_max) / 2).view(1, 3)
            scale = (pc_max - pc_min).max().reshape(1, 1) / 2
        else: # No scaling
            shift = torch.zeros([1, 3])
            scale = torch.ones([1, 1])

        # Apply the transformation
        pc = (pc - shift) / scale

        data = {
            'pointcloud': pc,
            'cate': cate_name,
            'id': idx, # Use index as a unique ID for this split
            'shift': shift,
            'scale': scale
        }

        # Apply any optional external transforms
        if self.transform is not None:
            data = self.transform(data)

        return data
### old collate
# def collate_fn_pad_point_clouds(batch):
#     """
#     Pads point cloud tensors to the largest number of points in the batch.
#     """
#     # Find the largest number of points in the current batch
#     max_num_points = max(p['pointcloud'].shape[0] for p in batch)
    
#     # Pad all point clouds to max_num_points
#     padded_batch = []
#     for data in batch:
#         point_cloud = data['pointcloud']
#         num_points_to_pad = max_num_points - point_cloud.shape[0]
#         # Pad with zeros
#         padding = torch.zeros((num_points_to_pad, 3), dtype=point_cloud.dtype)
#         padded_point_cloud = torch.cat([point_cloud, padding], dim=0)
#         data['pointcloud'] = padded_point_cloud
#         padded_batch.append(padded_point_cloud)
#     # Stack the padded point clouds
#     return torch.stack(padded_batch, dim=0)


# if __name__=="__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description='ShapeNetCore Dataset Example')
#     parser.add_argument('--path', type=str, required=True, help='Path to the ShapeNetCore dataset root directory')
#     parser.add_argument('--cates', type=str, nargs='+', default=['Airplane','Bag', 'Basket'], help='List of categories to load')
#     parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], default='train', help='Dataset split to use')
#     parser.add_argument('--scale_mode', type=str, choices=['global_unit', 'shape_unit', 'shape_bbox', 'shape_half', None], default='shape_unit', help='Scaling mode for point clouds')
    
#     args = parser.parse_args()
#     #args.path = f"/home/carlos/Rnet_local/datasets/shapenetCore"
#     dataset = ShapeNetCore(path=args.path, cates=args.cates, split=args.split, scale_mode=args.scale_mode)
#     print(f'Dataset size: {len(dataset)}')
#     sample = dataset[0]
#     print(sample)