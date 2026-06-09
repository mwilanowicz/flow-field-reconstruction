"""
Module: data_loader.py
Description: PyTorch data pipeline for the incompressible Navier-Stokes cylinder wake dataset.
Provides iterable batches of non-dimensional flow fields (u, v, p) spatial reconstruction
and dimensionality reduction.
Author: Marcel Wilanowicz
Date: 2026-06-05
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os

class CylinderDataset(Dataset):
    """
    Dataset for flow past a cylinder at Re=100.
    Provides snapshots of the fluid domain for spatial feature extraction 
    and autoencoder-based reconstruction.
    
    Reference: [Raissi et al., 2019, Sec. 4.1.1, p. 693]
    """

    def __init__(self, data_path):
        data = np.load(data_path)
        self.data = torch.from_numpy(data).float() # PyTorch tensor conversion
            
    def __len__(self):
        return len(self.data) # Total available snapshots in the dataset
    
    def __getitem__(self, idx):
        """
        Returns a pair of identical snapshots serving as both the input and 
        the target for the reconstruction loss function (Unsupervised Learning).
        """
        state = self.data[idx] # Flow field at time t^n

        return state, state
    
def data_loaders(data_path, batch_size=8):
    """
    Create sequential train, validation and test data loaders to evaluate 
    the model's performance across different time intervals of the vortex street.
    """
    dataset = CylinderDataset(data_path)

    # Sequential split to maintain physical time continuity
    train_idx = list(range(0, 160)) # 80% (160 samples)
    val_idx = list(range(160, 180)) # 10% (20 samples)
    test_idx = list(range(180, 200)) # 10% (20 samples)

    # Shuffling enabled only for training to improve convergence
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# Testing the script
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(
        current_dir, "../../data/processed/cylinder_nektar_wake_norm.npy"
    )

    try:
        train_loader, val_loader, test_loader = data_loaders(DATA_PATH, batch_size=8)
        print(f"Success: Loaders initialized. Train batches: {len(train_loader)}")

        # Fetching a single batch to verify shapes
        input_batch, target_batch = next(iter(train_loader))
        print(f"Batch shape: {input_batch.shape}")
        print(f"Value range: [{input_batch.min():.4f}, {input_batch.max():.4f}]")

        # Verifying identity mapping for autoencoder training
        identity_check = torch.allclose(input_batch, target_batch)
        print(f"Identity check (Input == Target for CAE loss): {identity_check}")

        if not identity_check:
            print("Warning: Input and target batches do not match.")

    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATA_PATH}.")