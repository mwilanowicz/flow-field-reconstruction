"""
File: engine.py
Description: Process logic for training and evaluaion of the model. 
Optimized for autoencoder regression.
Author: Marcel Wilanowicz
Date: 2026-05-13
"""

import torch

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, _ in loader:
        x = x.to(device)

        optimizer.zero_grad()
        output = model(x)

        target = x

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    return total_loss / total_samples

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    loss_u, loss_v, loss_p = 0.0, 0.0, 0.0
    total_samples = 0

    for x, _ in loader:
        x = x.to(device)
        output = model(x)

        target = x

        loss = criterion(output, target)
        total_loss += loss.item() * x.size(0)

        loss_u += criterion(output[:, 0], target[:, 0]).item() * x.size(0)
        loss_v += criterion(output[:, 1], target[:, 1]).item() * x.size(0)
        loss_p += criterion(output[:, 2], target[:, 2]).item() * x.size(0)

        total_samples += x.size(0)

    return (total_loss / total_samples, 
            loss_u / total_samples, 
            loss_v / total_samples, 
            loss_p / total_samples)

# Testing the script
if __name__ == "__main__":
    test_model = torch.nn.Conv2d(3, 3, kernel_size=1)
    test_device = torch.device("cpu")

    # Data simulation: (input_t, target_t+1)
    dummy_data = [(torch.randn(4, 3, 128, 256), torch.randn(4, 3, 128, 256)) for _ in range(2)]

    test_criterion = torch.nn.MSELoss()
    test_optimizer = torch.optim.Adam(test_model.parameters(), lr=0.001)

    try:
        print("--- Testing Reconstruction ---")
        train_loss_rec = train_one_epoch(test_model, dummy_data, test_optimizer, test_criterion, test_device)
        print(f"Train Loss: {train_loss_rec:.6f}")
        val_loss_rec, val_loss_u_rec, val_loss_v_rec, val_loss_p_rec = evaluate(test_model, dummy_data, test_criterion, test_device)
        print(f"Val Loss: {val_loss_rec:.6f} | u: {val_loss_u_rec:.6f}, v: {val_loss_v_rec:.6f}, p: {val_loss_p_rec:.6f}")

    except Exception as e:
        print(f"Engine test failed: {e}")
        