# Computer Vision for Fluid Dynamics: Baseline vs. PiML CAE in Flow Field Regression

## Project Overview
This project focuses on the reconstruction and prediction of fluid flow fields-specifically **Von Karman vortex street**-using deep learning. The objective is to evaluate performance of a standard Convolutional Autoencoder (CAE) against a Physics-Informed Machine Learning (PiML) approach.

## Problem Definition
The project is formulated as a **Computer Vision (CV) task**, specifically a **dense pixel-wise regression** on structured 2D grids. While architecturally similar to semantic segmentation, the objective is to predict continuous physical quantities rather than discrete class labels.

* **Inputs/Targets**: Velocity components ($u$, $v$) and Pressure ($p$).
* **Domain:** 2D incompressible flow past a circular cylinder at $Re = 100$.
* **Data Structure:** Uniform $128 \times 256$ grid interpolated from Nektar++ simulation snapshots.

### Physics Constraint
For the PiML model, we enforce the **Continuity Equation**, which ensures mass conservation (no fluid is created or destroyed). This acts as a domain-specific regularizer that narrows the network's hypothesis space:

$$\nabla \cdot \mathbf{u} = \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0,$$

where $\mathbf{u} = (u, v)$ is velocity field.

The regularization term is incorporated into the loss function using automatic differentiation:

$$L_\text{phys} = || \frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} ||^2$$

## Methodology
1. **Baseline CAE:** A standard Convolutional Autoencoder optimized using Mean Squared Error (MSE).
2. **PiML CAE:** An augmented architecture incorporating physics-based regularization via a composite loss function ($L = L_{\text{MSE}} + \lambda L_{\text{phys}}$).
3. **Analysis:** Comparative study of reconstruction fidelity (MSE) and physical consistency (divergence error), aimed at identifying limitations and potential improvements.

## Literature
* Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. [Link](https://doi.org/10.1016/j.jcp.2018.10.045)
* Murata, T., Fukami, K., & Fukagata, K. (2020). Nonlinear mode decomposition with convolutional neural networks for fluid dynamics. [Link](https://doi.org/10.1017/jfm.2019.822)
* Lusch, B., Kutz, J. N., & Brunton, S. L. (2018). Deep learning for universal linear embeddings. [Link](https://doi.org/10.1038/s41467-018-07210-0)

## Dataset:
The dataset is derived from the high-fidelity numerical simulations provided by Maziar Raissi:
[PINNs Data - Cylinder Flow](https://github.com/maziarraissi/PINNs/tree/master/main/Data)

## Technical Stack
* Language: Python 3.10+
* Frameworks: PyTorch
* Libraries: NumPy, SciPy, Matplotlib