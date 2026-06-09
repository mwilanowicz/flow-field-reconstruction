"""
File: train_cae.py
Description: Training script for reconstruction task using CAE. Orchestrates the loading of data, model 
initialization, and execution of the training loop for all models.
Author: Marcel Wilanowicz
Date: 2026-06-09
"""

import torch
import os
from models.cae_extended import CAE
from utils.engine import train_one_epoch, evaluate
from utils.data_loader import data_loaders
import random
import numpy as np

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# For GPUs
torch.backends.cudnn.deterministic = True # Only deterministic convolution algorithms
torch.backends.cudnn.benchmark = False # We don't want the fastest algorithm, only truth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_PATH = "../data/processed/cylinder_nektar_wake_norm.npy"

SAVE_PATH = "../checkpoints/cae_best.pth"
os.makedirs("../checkpoints", exist_ok=True)

HISTORY_PATH = "../history/cae_history.pth"
os.makedirs("../history", exist_ok=True)

train_loader, val_loader, test_loader = data_loaders(data_path=DATA_PATH, batch_size=8)

model = CAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

history = {
    'train_loss': [],
    'val_loss': [],
    'val_loss_u': [],
    'val_loss_v': [],
    'val_loss_p': []
}

best_val_loss = float('inf')
best_val_u = float('inf')
best_val_v = float('inf')
best_val_p = float('inf')
epochs = 100

print(110 * "-")
print("CAE Reconstruction Training:")

for epoch in range(epochs):
    msg = ""

    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_u, val_v, val_p = evaluate(model, val_loader, criterion, device)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_loss_u'].append(val_u)
    history['val_loss_v'].append(val_v)
    history['val_loss_p'].append(val_p)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_val_u = val_u
        best_val_v = val_v
        best_val_p = val_p
        torch.save(model.state_dict(), SAVE_PATH)
        msg = " - Model saved."
    
    print(f"Epoch [{epoch+1}/{epochs}] | Train MSE: {train_loss:.6f} | Val MSE: {val_loss:.6f} (u: {val_u:.6f}, v: {val_v:.6f}, p: {val_p:.6f}) {msg}")
        
torch.save(history, HISTORY_PATH)
print(f"\nCAE Training complete. Best Val MSE: {best_val_loss:.6f} (u: {best_val_u:.6f}, v: {best_val_v:.6f}, p: {best_val_p:.6f})")
