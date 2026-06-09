# Flow Field Reconstruction: Convolutional Autoencoder vs. PCA

## Project Overview
This project focuses on the spatial reconstruction of fluid flow fields, specifically the **Von Kármán vortex street**, using unsupervised machine learning and deep learning techniques. The objective is to evaluate and compare the performance of a non-linear Convolutional Autoencoder (CAE) against a traditional linear dimensionality reduction baseline (PCA / POD).

## Problem Definition
The project is formulated as a **dense pixel-wise regression** task on structured 2D grids. The models are designed to map high-dimensional input flow states into a low-dimensional latent space and reconstruct the continuous output fields.

* **Task:** Pixel-wise spatial reconstruction of flow field components ($u, v, p$) from a low-dimensional embedding.
* **Inputs/Targets:** Self-supervised reconstruction of Velocity components ($u, v$) and Pressure ($p$) snapshots at identical time steps (input is used as ground truth).
* **Domain:** 2D incompressible laminar flow past a circular cylinder at $Re = 100$.
* **Data Structure:** Uniform $128 \times 256$ spatial grid tensors with shape ($C, H, W$), where $C=3$.

## Methodology
1. **PCA (Baseline):** A classical linear method (Proper Orthogonal Decomposition) used to reduce data dimensionality. It reconstructs the flow fields by linearly combining a truncated set of dominant spatial modes and temporal coefficients.
2. **CAE:** A deep non-linear convolutional architecture optimized using Mean Squared Error (MSE) to capture complex, multi-scale spatial structures and non-linear interactions within an identical latent dimension.
3. **Analysis:** Comparison of reconstruction fidelity (Absolute Error maps), energy loss representation, and evaluation of latent space trajectory/clustering properties using t-SNE and PCA projections.

## Literature
* Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707. [Link](https://www.sciencedirect.com/science/article/pii/S0021999118307125?via%3Dihub)
* Murata, T., Fukami, K., & Fukagata, K. (2020). Nonlinear mode decomposition with convolutional neural networks for fluid dynamics. *Journal of Fluid Mechanics*, 882. [Link](https://doi.org/10.1017/jfm.2019.822)
* Brunton, S. L., Noack, B. R., & Koumoutsakos, P. (2020). Machine Learning for Fluid Mechanics. *Annual Review of Fluid Mechanics*, 52, 477-508. [Link](https://doi.org/10.1146/annurev-fluid-010719-060214)

## Dataset
The dataset is derived from numerical Navier-Stokes simulations provided by M. Raissi:
[PINNs Data - Cylinder Flow](https://github.com/maziarraissi/PINNs/tree/master/main/Data)

## Technical Stack
* **Language:** Python 3.10+
* **Framework:** PyTorch
* **Libraries:** Scikit-Learn, NumPy, SciPy, Matplotlib
