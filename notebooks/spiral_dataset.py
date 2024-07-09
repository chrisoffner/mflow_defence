from typing import Tuple
import torch

def spiral_dataset(
        n_samples: int,
        noise: float,
        n_rotations: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a dataset of two interlocking spirals

    Parameters
    ----------
    n_rotations : float
        Number of rotations performed by the spirals. Can be fractional values.
    n_samples : int
        Number of sample points to generate
    noise : float
        The degree of noise added to the data points.

    Returns
    -------
    Tuple[torch.Tensor, torch.LongTensor]
        (n_samples, 2) tensor of 2D data points and (n_samples,) tensor of labels
    """
    n_samples //= 2
    
    # Generate uniformly spaced angles
    theta = torch.sqrt(torch.linspace(0, 1, n_samples)) * n_rotations * 2 * torch.pi
    
    # Generate radius that grows linearly with the angle
    r = theta / (2 * torch.pi)
    
    # Convert polar coordinates to Cartesian coordinates
    noise *= 0.01
    d1x = r * torch.cos(theta) + torch.rand(n_samples) * noise
    d1y = r * torch.sin(theta) + torch.rand(n_samples) * noise
    
    points = torch.cat((torch.stack((d1x, d1y), dim=1),
                        torch.stack((-d1x, -d1y), dim=1)), dim=0)
    labels = torch.cat((torch.zeros(n_samples, dtype=torch.int64),
                        torch.ones(n_samples, dtype=torch.int64)))

    return points, labels
