"""
File: train_pca.py
Description: Training script for reconstruction task using PCA. Orchestrates the loading of data, model 
fitting, and MSE evaluation.
Author: Marcel Wilanowicz
Date: 2026-06-09
"""

import os
from models.pca import FlowPCA
import random
import numpy as np
import pickle

SEED = 42

random.seed(SEED)
np.random.seed(SEED)


DATA_PATH = "../data/processed/cylinder_nektar_wake_norm.npy"

SAVE_PATH = "../checkpoints/pca_best.pkl"
os.makedirs("../checkpoints", exist_ok=True)

HISTORY_PATH = "../history/pca_history.pkl"
os.makedirs("../history", exist_ok=True)

data = np.load(DATA_PATH)

x_train = data[0:160] # 80% (160 samples)
x_val = data[160:180] # 10% (20 samples)
x_test = data[180:200] # 10% (20 samples)

print(110 * "-")
print("PCA Baseline Decomposition & Evaluation:")

# Initialize and fit model (single analytical step)
model = FlowPCA(n_components=8)
print(f"Fitting PCA on {x_train.shape[0]} training samples...")
model.fit(x_train)

# Evaluate on training and validation sets
print(f"Evaluating reconstruction performance...")
x_train_pred = model(x_train)
x_val_pred = model(x_val)

train_loss = np.mean((x_train - x_train_pred) ** 2)
val_loss = np.mean((x_val - x_val_pred) ** 2)
val_u = np.mean((x_val[:, 0] - x_val_pred[:, 0]) ** 2)
val_v = np.mean((x_val[:, 1] - x_val_pred[:, 1]) ** 2)
val_p = np.mean((x_val[:, 2] - x_val_pred[:, 2]) ** 2)

# Storing single final scalars
history = {
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_loss_u': val_u,
    'val_loss_v': val_v,
    'val_loss_p': val_p
}

with open(SAVE_PATH, 'wb') as f:
    pickle.dump(model, f)

with open(HISTORY_PATH, 'wb') as f:
    pickle.dump(history, f)

print(f"\nPCA Training complete. Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} (u: {val_u:.6f}, v: {val_v:.6f}, p: {val_p:.6f})")
