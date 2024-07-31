from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50


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

        self.upsample = transforms.Resize(224)

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

