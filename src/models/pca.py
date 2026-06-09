'''
File: pca.py
Description: Linear baseline implementation using Principal Component Analysis (PCA).
Features data flattening for channel/spatial dimensions and inverse transfromation for dense 
pixelwise reconstruction, serving as a direct linear counterpart to the CAE.
Author: Marcel Wilanowicz
Date: 2026-06-04
'''

import numpy as np
from sklearn.decomposition import IncrementalPCA

class FlowPCA:
    """
    Linear PCA baseline for mapping high-dimensional flow fields into low-dimensional latent space.
    Matches the identical latent dimension (n_components) and input/output structure of the CAE.
    """
    def __init__(self, n_components=8):
        self.pca = IncrementalPCA(n_components=n_components)
        self.n_components = n_components
        self.original_shape = (3, 128, 256)
        self.flat_dim = 3 * 128 * 256

    def fit(self, x):
        """
        Expects input x as a NumPy array of shape (N, C, H, W)
        Flattens data to (N, C * H * W) and fits the PCA model.
        """
        n_samples = x.shape[0]
        x_flat = x.reshape(n_samples, self.flat_dim)
        self.pca.fit(x_flat)

    def encode(self, x):
        """
        Compresses high-dimensional flow fields into latent variables z.
        Input: (N, C, H, W) -> Output: (N, n_components)
        """
        n_samples = x.shape[0]
        x_flat = x.reshape(n_samples, self.flat_dim)
        return self.pca.transform(x_flat)
    
    def decode(self, z):
        """
        Reconstructs the full continous flow fields from the latent variables z.
        Input: (N, n_components) -> Output: (N, C, H, W)
        """
        n_samples = z.shape[0]
        x_reconstructed_flat = self.pca.inverse_transform(z)
        return x_reconstructed_flat.reshape(n_samples, *self.original_shape)
    
    def __call__(self, x):
        """
        Simulates the forward pass of a PyTorch module for easy integration in evaluation loop.
        """
        z = self.encode(x)
        return self.decode(z)
    
# Testing the script
if __name__ == "__main__":
    # Simulate a batch of 8 flow fields (N, C, H, W)
    dummy_input = np.random.randn(8, 3, 128, 256)

    try:
        model = FlowPCA(n_components=8)

        model.fit(dummy_input)
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")

        if dummy_input.shape == output.shape:
            print("Success: input shape equals output shape")
        else:
            print("Error: In/Out Dimensions differ.")

        print(f"Total extracted linear modes (latent dimensions): {model.n_components}")

    except Exception as e:
        print(f"An error occurred during PCA execuion: {e}")