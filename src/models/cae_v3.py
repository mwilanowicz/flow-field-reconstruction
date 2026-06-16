"""
File: cae_v3.py
Description: Enhanced Convolutional Autoencoder (CAE) architecture.
Features a higher capacity channel-wise scaling and limited spatial downsampling 
to retain fine-grained flow structures for high-dimensional reconstruction.
Author: Marcel Wilanowicz
Date: 2026-06-12
"""

import torch.nn as nn
import torch

class ConvBlock(nn.Module):
    """
    Building block consiting of Convolution, Batch Normalization and Tanh activation layers.
    """
    def __init__(self, input_channel, output_channel, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.Tanh()
        )

    def forward(self, x):
        return self.block(x)
    
class CAE(nn.Module):
    """
    CNN Autoencoder for mapping high-dimensional flow fields into low-dimensional latent space.
    """
    def __init__(self, latent_dim=16):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(3, 32, stride=2), # 3 x 128 x 256 -> 32 x 64 x 128 (spatial downsampling via stride)
            ConvBlock(32, 64, stride=2), # 32 x 64 x 128 -> 64 x 32 x 64 (spatial downsampling via stride)
            ConvBlock(64, 128, stride=1), # 32 x 32 x 64 -> 128 x 32 x 64 (spatial downsampling via stride)
            nn.Flatten(), # 128 x 32 x 64 = 128 * 32 * 64 = 262 144 values (flattening 3D tensor to 1D vector)

            # Compression into latent space z: 262 144 -> 16
            nn.Linear(128 * 32 * 64, latent_dim) 
        )

        self.decoder = nn.Sequential(
            # Decompression from latent space z: 16 -> 262 144
            nn.Linear(latent_dim, 128 * 32 * 64),
            nn.Unflatten(1, (128, 32, 64)), # Reshaping 1D vector back to 3D tensor: 128 x 32 x 64 (Channels, Height, Width)

            nn.Upsample(scale_factor=2, mode='nearest'), # 128 x 32 x 64  -> 128 x 64 x 128 (spatial restoration)
            ConvBlock(128, 64), # 128 x 64 x 128 -> 64 x 64 x 128  (channel reduction)

            nn.Upsample(scale_factor=2, mode='nearest'), # 64 x 64 x 128  -> 64 x 128 x 256 (spatial restoration)
            ConvBlock(64, 32), # 64 x 128 x 256 -> 32 x 128 x 256 (channel reduction)
             
            # Final linear output: no Tanh/BatchNorm to allow full range of physical values
            nn.Conv2d(32, 3, kernel_size=3, padding=1) # 32 x 128 x 256 -> 3 x 128 x 256 (32 hidden features to 3 output channels: u, v, p)
        )

    def forward(self, x):
        z = self.encoder(x) # Data goes into encoder resulting in latent space z
        output = self.decoder(z) # Data goes out from decoder, resulting in reconstructed image
        return output
    
# Testing the script
if __name__ == "__main__":
    LATENT_DIM = 16
    model = CAE(latent_dim=LATENT_DIM)
    dummy_input = torch.randn(8, 3, 128, 256)

    try:
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")

        if dummy_input.shape == output.shape:
            print("Succes: input shape equals output shape")
        
        else:
            print("Error: In/Out Dimensions differ.")

        total_params = 0
        for p in model.parameters():
            if p.requires_grad:
                total_params += p.numel()
        
        print(f"Total trainable parameters: {total_params:,}")

    except Exception as e:
        print(f"An error occurred during forward pass: {e}")
