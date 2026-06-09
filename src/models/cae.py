'''
File: cae.py
Description: Implementation of the Convolutional Autoencoder (CAE). Features a 
symmetrical architecture for spatial feature extraction and dense pixelwise reconstruction.
Author: Marcel Wilanowicz
Date: 2026-05-12
'''

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
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            ConvBlock(3, 16, stride=2), # 3 x 128 x 256 -> 16 x 64 x 128 (downsampling via stride)
            ConvBlock(16, 32, stride=2), # 16 x 64 x 128 -> 32 x 32 x 64
            nn.Flatten(), # 32 x 32 x 64 = 32 * 32 * 64 = 65536 values (flattening 3D tensor to 1D vector)

            # Compression into latent space z: 65536 -> 16
            nn.Linear(32 * 32 * 64, 16) 
        )

        self.decoder = nn.Sequential(
            # Decompression from latent space z: 16 -> 65536
            nn.Linear(16, 32 * 32 * 64),
            nn.Unflatten(1, (32, 32, 64)), # Reshaping values back to tensor (Channels, Height, Width)

            nn.Upsample(scale_factor=2, mode='nearest'), # 32 x 32 x 64 -> 32 x 64 x 128 (pixel doubling)
            ConvBlock(32, 16), # Processing features: 32 x 64 x 128 -> 16 x 64 x 128 (32 channels down to 16)

            nn.Upsample(scale_factor=2, mode='nearest'), # 16 x 64 x 128 -> 16 x 128 x 256 (final spatial restoration)
            nn.Conv2d(16, 3, kernel_size=3, padding=1) # 16 x 128 x 256 -> 3 x 128 x 256 (16 hidden features to 3 output channels: u, v, p)
            # Final linear output: no Tanh/BatchNorm to allow full range of physical values
        )

    def forward(self, x):
        z = self.encoder(x) # Data goes into encoder resulting in latent space z
        output = self.decoder(z) # Data goes out from dekoder, resulting in reconstructed image
        return output
    
# Testing the script
if __name__ == "__main__":
    model = CAE()
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
