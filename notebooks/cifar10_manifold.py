import torch
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split

from pathlib import Path
# from tqdm import tqdm

from manifold_flow.flows import ManifoldFlow
from manifold_flow import transforms, training
from manifold_flow.architectures.vector_transforms import create_vector_transform
from manifold_flow.architectures.image_transforms import create_image_transform

import logging
import time



def setup_logging():
    """Sets up logging."""
    # Set up Python's loggxing system to show INFO logs for M-Flow code
    logging.basicConfig(
        format="%(asctime)-5.5s %(message)s",
        datefmt="%H:%M",
        level=logging.INFO
    )

    logger = logging.getLogger(__name__)

    # Ensure that logging output of all other modules (e.g. matplotlib) is shown
    # only if it is at least a WARNING, not just an INFO.
    for key in logging.Logger.manager.loggerDict:
        if "experiments" not in key and "manifold_flow" not in key:
            logging.getLogger(key).setLevel(logging.WARNING)

def get_cifar_dataset(n_samples):
    """Creates a tensor dataset of n_samples from cifar10."""
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

    # Extract images from datasets
    train_images = train_dataset.data / 255.0  # Normalize to [0, 1]
    test_images = test_dataset.data / 255.0

    # Combine the images
    images = torch.cat([torch.tensor(train_images), torch.tensor(test_images)], dim=0).permute(0, 3, 1, 2)[:n_samples]
    print(f"Preparing {n_samples} images.")

    dataset = TensorDataset(images)
    return dataset



# Main function
def main():
    start_time = time.time()  # Record the start time
    setup_logging()

    dataset = get_cifar_dataset(n_samples = 1000)

    latentdim=2,
    outerlayers=20
    innerlayers=6
    levels=4
    splinebins=11
    splinerange=10.0
    dropout=0.0
    actnorm=True
    batchnorm=False
    contextfeatures=None
    linlayers=2
    linchannelfactor=1
    lineartransform="lu"

    steps_per_level = outerlayers // levels

    # Define M-Flow model
    params = {
        "base_transform_type": "rq-coupling",
        "n_flow_steps"       : 16,   # Depth (#layers)  of the "outer transform"
        "hidden_features"    : 100,  # Width (#neurons) of the "outer transform"
        "n_transform_blocks" : 6,     # ???
        "dropout"            : 0.
    }

    mflow = ManifoldFlow(
        data_dim=(3, 32, 32),
        latent_dim=25,
        inner_transform = create_vector_transform(
            dim = 25,
            flow_steps = 6 ,
            linear_transform_type="permutation",
            base_transform_type="rq-coupling",
            hidden_features=100,
            num_transform_blocks=2,
            dropout_probability=0.0,
            use_batch_norm=False,
            num_bins=8,
            tail_bound=3,
            apply_unconditional_transform=False,
            context_features=None,
        ),
        outer_transform=create_image_transform(
            3,
            32,
            32,
            levels=4,
            hidden_channels=100,
            steps_per_level=steps_per_level,
            num_res_blocks=2,
            alpha=0.05,
            num_bits=8,
            preprocessing="glow",
            dropout_prob=dropout,
            multi_scale=True,
            num_bins=splinebins,
            tail_bound=splinerange,
            postprocessing="partial_mlp",
            postprocessing_layers=linlayers,
            postprocessing_channel_factor=linchannelfactor,
            use_actnorm=actnorm,
            use_batchnorm=batchnorm,
        )
    )

    train    = True  #  Switch between training and loading the last trained model
    n_epochs = 5

    if train:
        trainer     = training.ForwardTrainer(mflow, run_on_gpu = True)
        metatrainer = training.AlternatingTrainer(mflow, trainer, trainer, run_on_gpu = True)
        losses_train, losses_eval = metatrainer.train(
            dataset,
            loss_functions=[training.losses.mse, training.losses.nll],
            loss_function_trainers=[0, 1],
            loss_labels=["MSE", "NLL"],
            loss_weights=[100., 1.],
            epochs=n_epochs,
            parameters=[mflow.parameters(), mflow.inner_transform.parameters()],
            verbose="all",
            trainer_kwargs=[
                {"forward_kwargs": {"mode": "projection"}},
                {"forward_kwargs": {"mode": "pie"}}
            ],
            write_per_epoch_plots=False,  # Writes sample plot to disk after each epoch
            params=params
        )
        torch.save(mflow.state_dict(), "../data/models/mflow_cifar10.pt")
        print(f"Model saved!")
    else:
        mflow.load_state_dict(torch.load("../data/models/mflow_cifar10.pt"))
        print(f"Model loaded!")

    end_time = time.time()  # Record the end time
    print(f"Total runtime: {end_time - start_time:.2f} seconds")

    


if __name__ == "__main__":
    main()