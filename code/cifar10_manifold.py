import torch
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split, Subset
from pathlib import Path
# from tqdm import tqdm

from manifold_flow.flows import ManifoldFlow
from manifold_flow import training
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

def get_cifar_dataset(n_samples = None):
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    """Creates a tensor dataset of n_samples from cifar10."""
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    # test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    if n_samples is not None: 
        # Create a subset of the first n_samples
        train_dataset = Subset(train_dataset, indices=range(n_samples))

    # Combine the images
    # images = torch.cat([torch.tensor(train_images), torch.tensor(test_images)], dim=0).permute(0, 3, 1, 2)[:n_samples]
    # print(f"Preparing {n_samples} images.")

    print(f'Total number of samples in the dataset: {len(train_dataset)}')

    # dataset = TensorDataset(images)
    return train_dataset



# Main function
def main():

    # Dimensions we tried
    # dims = [   2,   80,  159,  238,  316,  395,  474,  553,  631,  710,  789,
    #     867,  946, 1025, 1104, 1182, 1261, 1340, 1418, 1497, 1576, 1655,
    #    1733, 1812, 1891, 1969, 2048, 2127, 2206, 2284, 2363, 2442, 2520,
    #    2599, 2678, 2757, 2835, 2914, 2993, 3071]
    d = 500
    for i in range(8):
        print(f"Training mflows with {d}-dimensional manifold.")
        start_time = time.time()  # Record the start time
        setup_logging()

        # n_samples = 5000
        dataset = get_cifar_dataset() # max number is 50.000

        latentdim = d
        outerlayers = 20
        innerlayers = 6
        levels = 4
        splinebins = 11
        splinerange = 10.0
        dropout = 0.0
        actnorm = True
        batchnorm = False
        contextfeatures = None
        linlayers = 2
        linchannelfactor = 1
        lineartransform = "lu"

        steps_per_level = outerlayers // levels

        # Define M-Flow model
        params = {
            "base_transform_type": "rq-coupling",
            "n_flow_steps"       : 16,   # Depth (#layers)  of the "outer transform"
            "hidden_features"    : 100,  # Width (#neurons) of the "outer transform"
            "n_transform_blocks" : 6,     # ???
        }

        mflow = ManifoldFlow(
            data_dim=(3, 32, 32),
            latent_dim=latentdim,
            inner_transform = create_vector_transform(
                dim = latentdim,
                flow_steps = 16 , # previously 6
                linear_transform_type = "permutation",
                base_transform_type = "rq-coupling",
                hidden_features = 100,
                num_transform_blocks = 6, #previously 2
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
                postprocessing="partial_nsf",
                postprocessing_layers=linlayers,
                postprocessing_channel_factor=linchannelfactor,
                use_actnorm=actnorm,
                use_batchnorm=batchnorm,
            )
        )

        train    = True  #  Switch between training and loading the last trained model
        n_epochs = 25

        if train:
            trainer     = training.ForwardTrainer(mflow, run_on_gpu = True)
            metatrainer = training.AlternatingTrainer(mflow, trainer, trainer, run_on_gpu = True)
            losses_train, losses_eval = metatrainer.train(
                dataset,
                batch_sizes= 256,
                loss_functions=[training.losses.mse, training.losses.nll],
                loss_function_trainers=[0, 1],
                loss_labels=["MSE", "NLL"],
                loss_weights=[100., 0.01],
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
            torch.save(mflow.state_dict(), f"../data/models/mflow_cifar10_nsamples50000_normalized_dim{d}_run{i}.pt")
            print(f"Model saved!")
        else:
            mflow.load_state_dict(torch.load(f"../data/models/mflow_cifar10_nsamples50000_normalized_dim{d}_run{i}.pt"))
            print(f"Model loaded!")

        end_time = time.time()  # Record the end time
        total_runtime = end_time - start_time
        print(f"Total runtime: {total_runtime:.2f} seconds")

        print(f"Final MSE loss: {losses_eval[-1]}")
        torch.save(losses_eval, f"../data/models/mflow_cifar10_nsamples50000_normalized_dim{d}_run{i}_losses_eval.pt")
        torch.save(losses_train, f"../data/models/mflow_cifar10_nsamples50000_normalized_dim{d}_run{i}_losses_train.pt")

        # Write current d and runtime to a .txt file
        with open(f"dims_runtime_log_nsamples50000_normalized.txt", "a") as log_file:  # Open the file in append mode
            # log_file.write(f"Run: {i}, Dimension: {d}, Runtime: {total_runtime:.2f} seconds, Final eval loss: {losses_eval[-1]}\n")
            log_file.write(f"Dimension: {d}, Runtime: {total_runtime:.2f} seconds, Final eval loss: {losses_eval[-1]}\n")



if __name__ == "__main__":
    main()