import h5py
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import pickle


def pad_collate_fn(batch, max_particles=500):
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

        # Retrieve the data from the pre-loaded arrays
        shower = self.all_showers[idx]
        #Normalize energy between 0 and 1
        max_e = np.max(self.all_energies)
        min_e = np.min(self.all_energies)
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
