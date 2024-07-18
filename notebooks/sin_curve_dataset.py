from typing import Tuple
import torch

def single_sin_curve_dataset(
        n_samples: int,
        noise: float,
        amplitude: float,
        frequency: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a dataset based on a single sine curve with added noise, centers it around zero,
    and scales the data to be between 0 and 1.

    Parameters
    ----------
    n_samples : int
        Number of sample points to generate.
    noise : float
        The degree of noise added to the data points.
    amplitude : float
        Amplitude of the sine wave.
    frequency : float
        Frequency of the sine hard to maintain spatial relationship in label assignment.
        
    Returns
    -------
    Tuple[torch.Tensor, torch.LongTensor]
        (n_samples, 2) tensor of 2D data points centered around zero, scaled between 0 and 1,
        and (n_samples,) tensor of labels.
    """
    # Generate x values
    x = torch.linspace(0, 2 * torch.pi * frequency, n_samples)
    
    # Generate y values with noise
    y = amplitude * torch.sin(x) + torch.randn(n_samples) * noise
    
    # Center the x values around 0
    x_centered = x - x.mean()
    
    # Center the y values around 0
    y_centered = y - y.mean()

    # Scale the data to be between 0 and 1
    x_min, x_max = x_centered.min(), x_centered.max()
    y_min, y_max = y_centered.min(), y_centered.max()

    x_scaled = (x_centered) / (x_max - x_min)
    y_scaled = (y_centered) / (y_max - y_min)

    # Prepare dataset using scaled values
    points = torch.stack((x_scaled, y_scaled), dim=1)

    points[:,0] *= 4.6
    points[:,1] *= 3.0
    
    # Assign labels based on the original x value; each bump spans from 0 to 2*pi
    labels = torch.ceil(x / (2 * torch.pi)) % 2  # Alternating labels for successive bumps
    labels = labels.to(torch.int64)  # Convert labels to torch.int64

    return points, labels