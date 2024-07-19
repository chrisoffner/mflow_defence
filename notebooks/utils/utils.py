from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
from torchvision.models import resnet50

# from pathlib import Path


def get_grid_predictions(
        model: torch.nn.Module,
        probs: bool = False,
        grid_res: int = 500
    ) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    """
    Generates a grid of 2D points on which we will perform inference with the
    trained classifier. The resulting predicted labels will then be used to draw
    a contour plot (not in this function) showing the decision boundary.


    Parameters
    ----------
    model : torch.nn.Module
        Trained classifier.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, torch.Tensor]
        Grid point x-values, y-values, and model predictions for those points.
    """
    xs, ys = np.meshgrid(np.linspace(-2.3, 2.3, grid_res), np.linspace(-3, 3, grid_res))
    grid_points = np.column_stack((xs.ravel(), ys.ravel())).astype(np.float32)

    model.eval()
    with torch.no_grad():
        grid_preds = model(torch.from_numpy(grid_points))

    if probs:
        # Return just probabilities of being in class 1
        grid_preds = grid_preds[:, 1]
    else:
        # Turn probabilities into discrete class labels
        grid_preds = grid_preds.argmax(axis=1)

    return xs, ys, grid_preds


def create_contour_plot(
        xs: np.ndarray,
        ys: np.ndarray,
        grid_preds: torch.Tensor,
        dataset: TensorDataset,
        colormap,
        file_path: str|None = None, epoch: int|None = None
    ):
    """
    Creates a contour plot using the classifier predictions on a regular grid of
    points. This function can be called inside the training loop, in which case
    it expects `file_name` and `epoch` arguments, and will write the plot to
    disk. If these last two arguments are `None`, then the plot is just shown in
    this Jupyter notebook but not written to disk.

    Parameters
    ----------
    xs : np.ndarray
        x-coordinates of grid points
    ys : np.ndarray
        y-coordinates of grid points
    grid_preds : torch.Tensor
        Label predictions for grid points
    file_name : str | None, optional
       Where the plot should be saved to disk, if at all. By default None
    epoch : int | None, optional
        Epoch, used as plot title, by default None
    """
    plt.figure(figsize=(5, 5), dpi=200)

    if epoch is not None:
        plt.title(f"Epoch {epoch}")
    plt.contourf(
        xs,
        ys,
        grid_preds.reshape(xs.shape),
        levels=2,  # For binary classification, we use 2 levels
        cmap=colormap, 
        alpha=0.3, 
        antialiased=True
    )
    X, Y = dataset.tensors
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=1, cmap=colormap)
    plt.xlim(X.T.min()*1.1, X.T.max()*1.1)
    plt.ylim(X.T.min()*1.1, X.T.max()*1.1)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.tight_layout()

    if file_path is not None:
        plt.savefig(file_path)
        
        # Clear memory
        plt.close()
        plt.clf()
        plt.cla()
        plt.close("all")

class resnet_CIFAR10(nn.Module):
    def __init__(self, resnet_checkpoint, device):
        super(resnet_CIFAR10, self).__init__()

        self.upsample = nn.Upsample(size=244)

        # Init ResNet50  
        rn = resnet50()
        # Modify the ResNet-50 architecture for CIFAR-10: 10 output classes 
        rn.fc = torch.nn.Linear(rn.fc.in_features, 10)
        # Load pre-trained weights
        rn.load_state_dict(torch.load(resnet_checkpoint, map_location=torch.device(device)))  
        self.resnet = rn
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.resnet(x)
        return x

# def plot_grid_lines(
#     mflow:     torch.nn.Module,
#     n_samples: int,
#     range:     float,
#     axis:      str = "both",
#     file_path: Path | None = None,
# ):
#     # Set up the plot
#     plt.figure(figsize=(5, 5), dpi=200)
#     plt.scatter(*manifold_points.T, s=1, alpha=1, c="plum")

#     grid_values = np.linspace(-range, range, 33)

#     for val in grid_values:
#         # Generate points on a line grid
#         if axis in ["x", "both"]:
#             line_points = np.column_stack([np.full(n_samples, val), np.linspace(-range, range, n_samples)])
#         if axis in ["y", "both"]:
#             line_points = np.column_stack([np.linspace(-x_range, x_range, n_samples), np.full(n_samples, val)])

#         # Convert to torch tensor and ensure float32 dtype
#         line_points_tensor = torch.tensor(line_points, dtype=torch.float32)

#         # Map grid points from latent space to ambient data space
#         points_proj = mflow.outer_transform.inverse(line_points_tensor)[0].detach().numpy()

#         # Plot the warped grid
#         color = "hotpink" if abs(val) < 0.1 else "lightgray"
#         lw    = 2         if abs(val) < 0.1 else 0.8
#         plt.plot(points_proj[:, 0], points_proj[:, 1], lw=lw, color=color)

#     # Finalize plot settings
#     plt.gca().set_aspect("equal", adjustable="box")
#     plt.axis("off")
#     plt.xlim(-2.3, 2.3)
#     plt.ylim(-2.3, 2.3)
#     plt.tight_layout()

#     if file_path is not None:
#         plt.savefig(str(file_path))

#     plt.close()
#     plt.cla()
#     plt.clf()