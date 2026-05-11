"""
Module: data_loader.py
Description: PyTorch data pipeline for the incompressible Navier-Stokes cylinder wake dataset.
Provides iterable batches of non-dimensional flow fields (u, v, p) for CNN/CAE architectures.
Author: Marcel Wilanowicz
Date: 2026-05-10
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

class CylinderDataset(Dataset):
    """
    Dataset for flow past a cylinder at Re=100.
    Maps snapshot at time t^n to t^{n+1}, conceptually aligning with the discrete time 
    framework for limited snapshots.
    
    Reference: [Raissi et al., 2019, Sec. 4.1.1, p. 693]
    """

    def __init__(self, data_path):
        data = np.load(data_path)
        self.data = torch.from_numpy(data).float() # PyTorch tensor conversion
            
    def __len__(self):
        return len(self.data) - 1 # Total frames minus one (last frame has no next state)
    
    def __getitem__(self, idx):
        """
        Returns a pair of snapshots representing the state transition u^n -> u^{n+1}
        Based on the Runge-Kutta formulation for limited snapshots [Raissi et al., 2019, p. 694].
        """
        curr_state = self.data[idx] # Flow field at time t^n (n-th step)
        next_state = self.data[idx + 1] # Flow field at time t^{n+1} (n-th + 1 step)

        return curr_state, next_state
    
def data_loaders(data_path, batch_size=8):
    """
    Create sequential train, validation and test data loaders for temporal continuity of the 
    Karman vortex street.
    """
    dataset = CylinderDataset(data_path)

    # Sequential split to maintain physical time continuity
    train_idx = list(range(0, 160)) # 80% (160 samples)
    val_idx = list(range(160, 180)) # 10% (20 samples)
    test_idx = list(range(180, 199)) # ~10% (19 samples)

    # Shuffling enabled only for training to improve convergence
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Sanity check for data loaders
if __name__ == "__main__":
    DATA_PATH = '../data/processed/cylinder_nektar_wake_norm.npy'

    try:
        train_loader, val_loader, test_loader = data_loaders(DATA_PATH, batch_size=8)
        print(f"Success: Loaders initialized. Train batches: {len(train_loader)}")

        curr_batch, next_batch = next(iter(train_loader))
        print(f"Batch shape: {curr_batch.shape}")
        print(f"Value range: [{curr_batch.min():.4f}, {curr_batch.max():.4f}]")

        dataset = CylinderDataset(DATA_PATH)

        # Testing specific data samples for temporal overlap
        curr_data, next_data = dataset[0] # u^0 (input), u^1 (target)
        curr_data1, next_data1 = dataset[1] # u^1 (input), u^2 (target)

        # Data should overlap between indices (target of u^n must be the input of u^{n+1})
        continuity_check = torch.allclose(next_data, curr_data1) 
        print(f"Temporal continuity (target u^1 == input u^1: {continuity_check})")

        if not continuity_check:
            print("Error: Temporal gap detected in dataset indexing.")

    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATA_PATH}.")
