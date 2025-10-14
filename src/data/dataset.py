import os
import json
import torch
from copy import copy
import h5py
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle
from typing import List, Tuple



class HDF5Dataset(Dataset):
    """
    A PyTorch Dataset for loading data from a single HDF5 file.
    It supports multiple groups within the HDF5 file, treating each
    entry as a separate data point.
    """

    def __init__(self, h5_path):
        """
        Initializes the dataset by creating an index map from a global
        index to the specific group and item within that group.

        Args:
            h5_path (str): The path to the HDF5 file.
        """
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found at: {h5_path}")
        
        self.h5_path = h5_path
        self.groups = []
        self.num_events_in_group = []
        self._load_index_map()

    def _load_index_map(self):
        """
        Walks through the HDF5 file to build an index map. This allows
        us to quickly find the correct group and dataset for any given
        global index.
        """
        with h5py.File(self.h5_path, 'r') as hf:
            for group_name in hf.keys():
                group = hf[group_name]
                # Assuming 'energies' is present in all groups and has a length
                # that represents the number of events in that group.
                if 'energies' in group:
                    num_events = len(group['energies'])
                    if num_events > 0:
                        self.groups.append(group_name)
                        self.num_events_in_group.append(num_events)
                        
        if not self.groups:
            raise ValueError("No data groups found in the HDF5 file.")
        
        # Calculate cumulative counts to map a global index to a specific group
        self.cumulative_counts = np.cumsum(self.num_events_in_group)
        self.total_events = self.cumulative_counts[-1]

    def __len__(self):
        """Returns the total number of events in the dataset."""
        return self.total_events

    def __getitem__(self, idx):
        """
        Retrieves a single data sample from the HDF5 file.

        Args:
            idx (int): The global index of the data point.

        Returns:
            A tuple of PyTorch tensors: (showers, energy, pid, gap_pid).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Find which group the global index belongs to
        group_idx = np.searchsorted(self.cumulative_counts, idx, side='right')
        
        # Calculate the local index within the determined group
        local_idx = idx - (self.cumulative_counts[group_idx-1] if group_idx > 0 else 0)
        group_name = self.groups[group_idx]

        with h5py.File(self.h5_path, 'r') as hf:
            group = hf[group_name]
            
            # Load the data for the specific index
            shower_data = group['showers'][local_idx]
            energy_data = group['energies'][local_idx]
            pid_data = group['pid'][()]
            gap_pid_data = group['gap_pid'][()]

        # Convert to PyTorch tensors and return
        shower = torch.from_numpy(shower_data).float()
        energy = torch.tensor(energy_data).float()
        pid = torch.tensor(pid_data).long()
        gap_pid = torch.tensor(gap_pid_data).long()

        return (shower, energy, pid, gap_pid)


# A helper dictionary to store the mapping from file index to global index range
file_index_to_global_range = {}

class PklDataset(Dataset):
    """
    A PyTorch Dataset for loading and serving data from a folder of pickle files.
    Each pickle file is expected to contain a dictionary with 'showers', 'particle_pid', 'gap_pid', 
    primary_energies'.
    """

    def __init__(self, data_dir, transform=None):
        """
        Initializes the dataset by loading all pickle files into memory.

        Args:
            data_dir (str): The path to the directory containing the .pkl files.
        """
        self.data_dir = data_dir
        self.all_showers = []
        self.all_energies = []
        self.all_pids = []
        self.all_gap_pids = []
        self._load_data()
        self.transform = transform

    def _load_data(self):
        """
        Loads all the data from the .pkl files in the specified directory.
        This approach loads the entire dataset into memory, which is suitable
        for a moderate number of files.
        """
        file_paths = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.pkl')]
        
        # We need to load all data to know the total number of events
        for file_path in file_paths:
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                # Extract data from the dictionary. Use .pop() to get and remove
                # the list from the dictionary.
                showers = data['showers'].pop()
                energies = data['energies'].pop()
                pid = data['pid'].pop()
                gap_pid = data['gap_pid'].pop() 
                # Add a sanity check to ensure the data is not empty and has matching lengths
                if showers.size > 0 and len(showers) == len(energies):
                    self.all_showers.append(showers)
                    self.all_energies.append(energies)
                    self.all_pids.append(np.full(len(showers), pid))  # Replicate PID for each shower
                    self.all_gap_pids.append(np.full(len(showers), gap_pid)) # Replicate GAP PID for each shower
                else:
                    print(f"Warning: Skipping file '{file_path}' due to inconsistent or empty data.")
            except (pickle.UnpicklingError, FileNotFoundError, KeyError, IndexError) as e:
                print(f"Error loading file '{file_path}': {e}")
        
        # Concatenate all numpy arrays into single, large arrays
        if self.all_showers:
            self.all_showers = np.concatenate(self.all_showers, axis=0)
            self.all_energies = np.concatenate(self.all_energies, axis=0)
            self.all_pids = np.concatenate(self.all_pids, axis=0)
            self.all_gap_pids = np.concatenate(self.all_gap_pids, axis=0)

    def __len__(self):
        """
        Returns the total number of events (showers) in the dataset.
        """
        return len(self.all_showers)

    def __getitem__(self, idx):
        """
        Retrieves a single data sample at the specified index.
        The data is returned as a tuple of PyTorch tensors.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #max_e = np.max(self.all_energies)
        #min_e = np.min(self.all_energies)
        #TODO read this from config file GenAi datageneration
        max_e = 1000000
        min_e = 1000
        # Retrieve the data from the pre-loaded arrays
        shower = self.all_showers[idx]
        #Normalize energy between 0 and 1
        energy = (self.all_energies[idx] - min_e) / (max_e-min_e)
        #energy = self.all_energies[idx]
        pid = self.all_pids[idx]
        gap_pid = self.all_gap_pids[idx]
        if self.transform:
            shower = self.transform(shower)

        # Convert numpy arrays to PyTorch tensors
        # The showers data has shape (N_particles, 4)
        shower = torch.from_numpy(shower).float()
        
        # The energy, pid, and gap_pid are scalars, convert to single-element tensors
        energy = torch.tensor(energy).float()
        pid = torch.tensor(pid).long()
        gap_pid = torch.tensor(gap_pid).long()

        # Return the tensors as a tuple.
        # This format is easy for the DataLoader to handle.
        return (shower, energy, pid, gap_pid)


class LazyPklDataset(Dataset):
    """
    A PyTorch Dataset that loads data from pickle files on-demand (lazily) to avoid
    loading the entire dataset into memory.
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # The map will store tuples: (file_path, index_within_file)
        # This list IS the only piece of data stored in memory for the whole dataset.
        self.global_index_map: List[Tuple[str, int]] = []
        
        self._create_global_index_map()

    def _create_global_index_map(self):
        """
        Scans all files to determine the total number of events and creates 
        a map from global index to (file_path, local_index). 
        This is the only necessary step that requires reading *some* metadata 
        from the files, but not the heavy data itself.
        """
        file_paths = [os.path.join(self.data_dir, f) 
                      for f in os.listdir(self.data_dir) if f.endswith('.pkl')]
        
        # We store and reuse the loaded file data temporarily
        for file_path in file_paths:
            try:
                with open(file_path, 'rb') as f:
                    # Load the file content
                    data = pickle.load(f)

                # Get the shower and energy arrays (assuming they are lists containing one array each)
                # NOTE: We only need the *length* of the arrays, not the array contents.
                showers = data['showers'][0]
                
                # Check if it's a list containing a numpy array, get its length
                num_showers = len(showers) 
                
                # Create the mapping for all events in this file
                for local_idx in range(num_showers):
                    self.global_index_map.append((file_path, local_idx))
                    
            except (pickle.UnpicklingError, FileNotFoundError, KeyError, IndexError) as e:
                print(f"Error reading file structure '{file_path}': {e}. Skipping file.")

        print(f"Dataset indexed. Total events found: {len(self.global_index_map)}")

    def __len__(self):
        """Returns the total number of events in the dataset."""
        return len(self.global_index_map)

    def __getitem__(self, idx):
        """Retrieves a single data sample by loading the necessary file on demand."""
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # 1. Look up the file and local index
        file_path, local_idx = self.global_index_map[idx]
        
        # 2. Load the entire file (This is the I/O-heavy step)
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        # Extract the necessary data arrays (assuming they are single-element lists)
        all_showers_in_file = data['showers'][0]
        all_energies_in_file = data['energies'][0]
        pid_in_file = data['pid'][0]
        gap_pid_in_file = data['gap_pid'][0]
        
        # 3. Retrieve the specific sample (the "lazy" part)
        shower = all_showers_in_file[local_idx]
        energy = all_energies_in_file[local_idx]
        pid = pid_in_file
        gap_pid = gap_pid_in_file

        # NOTE: Using hardcoded max/min is fine, but you might move it to a config
        max_e = 1000000
        min_e = 1000
        
        # Normalization and transformation
        energy = (energy - min_e) / (max_e - min_e)

        if self.transform:
            shower = self.transform(shower)

        # 4. Convert to PyTorch tensors
        shower = torch.from_numpy(shower).float()
        energy = torch.tensor(energy).float()
        pid = torch.tensor(pid).long()
        gap_pid = torch.tensor(gap_pid).long()

        return (shower, energy, pid, gap_pid)

# The provided dictionary mapping synset IDs to category names
synsetid_to_cate = {'02691156': 'Airplane', '02773838': 'Bag', '02801938': 'Basket', '02808440': 'Bathtub', '02818832': 'Bed', '02828884': 'Bench', '02876657': 'Bottle', '02880940': 'Bowl', 
                    '02924116': 'Bus', '02933112': 'Cabinet', '02747177': 'Can', '02942699': 'Camera', '02954340': 'Cap', '02958343': 'Car', '03001627': 'Chair', '03046257': 'Clock', 
                    '03207941': 'Dishwasher', '03211117': 'Monitor', '04379243': 'Table', '04401088': 'Telephone', '02946921': 'Tin_can', '04460130': 'Tower', '04468005': 'Train',
                    '03085013': 'Keyboard', '03261776': 'Earphone', '03325088': 'Faucet', '03337140': 'File', '03467517': 'Guitar', '03513137': 'Helmet', '03593516': 'Jar', '03624134': 'Knife', 
                    '03636649': 'Lamp', '03642806': 'Laptop', '03691459': 'Speaker', '03710193': 'Mailbox', '03759954': 'Microphone', '03761084': 'Microwave', '03790512': 'Motorcycle', '03797390': 'Mug',
                    '03928116': 'Piano', '03938244': 'Pillow', '03948459': 'Pistol', '03991062': 'Pot', '04004475': 'Printer', '04074963': 'Remote_control', '04090263': 'Rifle', '04099429': 'Rocket', 
                    '04225987': 'Skateboard', '04256520': 'Sofa', '04330267': 'Stove', '04530566': 'Vessel', '04554684': 'Washer', '02992529': 'Cellphone', '02843684': 'Birdhouse', '02871439': 'Bookshelf'}
cate_to_synsetid = {v: k for k, v in synsetid_to_cate.items()}

int_classes = {'Airplane': 0, 'Bag': 1, 'Basket': 2, 'Bathtub': 3, 'Bed': 4, 'Bench': 5, 'Bottle': 6, 
               'Bowl': 7, 'Bus': 8, 'Cabinet': 9, 'Can': 10, 'Camera': 11, 'Cap': 12, 'Car': 13, 
               'Chair': 14, 'Clock': 15, 'Dishwasher': 16, 'Monitor': 17, 'Table': 18, 'Telephone': 19, 
               'Tin_can': 20, 'Tower': 21, 'Train': 22, 'Keyboard': 23, 'Earphone': 24, 'Faucet': 25, 
               'File': 26, 'Guitar': 27, 'Helmet': 28, 'Jar': 29, 'Knife': 30, 'Lamp': 31, 'Laptop': 32, 
               'Speaker': 33, 'Mailbox': 34, 'Microphone': 35, 'Microwave': 36, 'Motorcycle': 37, 
               'Mug': 38, 'Piano': 39, 'Pillow': 40, 'Pistol': 41, 'Pot': 42, 'Printer': 43, 
               'Remote_control': 44, 'Rifle': 45, 'Rocket': 46, 'Skateboard': 47, 'Sofa': 48, 
               'Stove': 49, 'Vessel': 50, 'Washer': 51, 'Cellphone': 52, 'Birdhouse': 53, 'Bookshelf': 54}

class ShapeNetCore(Dataset):

    def __init__(self, path, cates, split, scale_mode, max_num_points = 3000, transform=None):
        super().__init__()
        assert isinstance(cates, list), '`cates` must be a list of cate names.'
        assert split in ('train', 'val', 'test')
        assert scale_mode in ('global_unit', 'shape_unit', 'shape_bbox', 'shape_half', 'shape_34', None)
        
        self.path = path  # The root directory of the dataset
        self.split = split
        self.scale_mode = scale_mode
        self.transform = transform
        self.stats = None
        self.max_num_points = max_num_points   
        
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
        # Define a fixed number of points for all point clouds
        num_points = self.max_num_points # You can adjust this value

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
            'X': pc,
            'y': int_classes[cate_name],
            'id': idx, # Use index as a unique ID for this split
            #'shift': shift,
            #'scale': scale
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


# if __name__ == '__main__':
#     # Define the path to your data directory
#     # NOTE: You must change this path to your actual directory
#     data_directory = '/pscratch/sd/c/ccardona/datasets/G4_individual_sims_pkl'
    
#     # Check if the directory exists before trying to create the dataset
#     if not os.path.isdir(data_directory):
#         print(f"Error: Directory '{data_directory}' not found.")
#     else:
#         print(f"Attempting to load data from: {data_directory}")

#         try:
#             # Create an instance of the dataset
#             dataset = PklDataset(data_directory)
#             print(f"Successfully loaded dataset with {len(dataset)} events.")

#             # Create a DataLoader to iterate over the dataset in batches
#             batch_size = 32
#             # Set num_workers > 0 for multi-process data loading, which can be faster.
#             # On Windows, you may need to set this to 0.
#             dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

#             # Iterate over the DataLoader to get batches of data
#             print(f"Iterating through the data in batches of {batch_size}...")
#             for i, (showers_batch, energies_batch, pids_batch, gap_pids_batch) in enumerate(dataloader):
#                 print(f"Batch {i+1}:")
#                 print(f"  Showers batch shape: {showers_batch.shape}")
#                 print(f"  Energies batch shape: {energies_batch.shape}")
#                 print(f"  PIDs batch shape: {pids_batch.shape}")
#                 print(f"  Gap PIDs batch shape: {gap_pids_batch.shape}")
                
#                 # We can break after a few batches for demonstration
#                 if i >= 2:
#                     break

#         except Exception as e:
#             print(f"An error occurred during dataset creation or loading: {e}")


# if __name__ == '__main__':
#     # Define the path to your combined HDF5 file
#     h5_file_path = '/pscratch/sd/c/ccardona/datasets/all_sims_combined.h5'

#     try:
#         # Create an instance of the HDF5 dataset
#         dataset = HDF5Dataset(h5_file_path)
#         print(f"Successfully loaded HDF5 dataset with {len(dataset)} total events.")
        
#         # --- Suggestion for splitting data ---
        
#         # Define the split ratios
#         train_ratio = 0.8
#         val_ratio = 0.1
#         test_ratio = 0.1

#         # Calculate the number of samples for each split
#         num_events = len(dataset)
#         num_train = int(num_events * train_ratio)
#         num_val = int(num_events * val_ratio)
#         num_test = num_events - num_train - num_val

#         # Use random_split to create the subsets
#         train_dataset, val_dataset, test_dataset = random_split(
#             dataset, [num_train, num_val, num_test]
#         )

#         print(f"\nDataset split into:")
#         print(f"  Training set: {len(train_dataset)} events")
#         print(f"  Validation set: {len(val_dataset)} events")
#         print(f"  Test set: {len(test_dataset)} events")

#         # Create DataLoaders for each subset
#         batch_size = 32
#         train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#         val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
#         test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

#         # Iterate over one batch from the training DataLoader to demonstrate
#         print("\nDemonstrating a batch from the training DataLoader...")
#         showers_batch, energies_batch, pids_batch, gap_pids_batch = next(iter(train_dataloader))
        
#         print(f"  Showers batch shape: {showers_batch.shape}")
#         print(f"  Energies batch shape: {energies_batch.shape}")
#         print(f"  PIDs batch shape: {pids_batch.shape}")
#         print(f"  Gap PIDs batch shape: {gap_pids_batch.shape}")

#     except Exception as e:
#         print(f"\nAn error occurred: {e}")
#         print("Please ensure the HDF5 file has been created correctly by the previous script.")
