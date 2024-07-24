import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from utils.attacks import fgsm, pgd
from utils.utils import resnet_CIFAR10

import os
import argparse
from pathlib import Path
import math

parser = argparse.ArgumentParser(description="Generate attacked CIFAR-10 datasets")
parser.add_argument('--resnet', '-r', type=Path, help='path to trained resnet checkpoint', required=True, default="../models/resnet/resnet50_cifar10.pt")

parser.add_argument('--num_eps', '-n', type=int, help='number of epsilons', required=False, default=50)
parser.add_argument('--min_eps', '-e', type=float, help='minimum value of attack epsilon', required=False, default=0.0)
parser.add_argument('--max_eps', '-E', type=float, help='maximum value of attack epsilon', required=False, default=0.1)
parser.add_argument('--eps_step', '-s', type=float, help='epsilon step size for PGD', required=False, default=1)

args = parser.parse_args()


print("==> running with arguments:")
print(f"   * resnet: {args.resnet}")
print(f"   * num_eps: {args.num_eps}")
print(f"   * min_eps: {args.min_eps}")
print(f"   * max_eps: {args.max_eps}")
print(f"   * eps_step: {args.eps_step}")

# Set device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# Define transform
transform = transforms.ToTensor()

# Load CIFAR-10 test dataset
bs = 64
test_data = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=True, transform=transform)
test_loader = DataLoader(test_data, batch_size=bs, shuffle=False)
print(f"dataset size: {len(test_data)}")

# Validate ResNet checkpoint path
assert os.path.exists(args.resnet), 'Error: no checkpoint file found!'

# Init modified ResNet-50
model = resnet_CIFAR10(args.resnet, device)

model.to(device)
model.eval()

# Define CIFAR-10 classes
classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# define range of epsilons to try attacks with
n_epsilons = args.num_eps
min_eps = args.min_eps
max_eps = args.max_eps
epsilons = torch.linspace(min_eps, max_eps, n_epsilons)
steps = torch.tensor([math.floor(min(255 * eps + 4, 1.25 * 255 * eps)) for eps in epsilons])
eps_step = args.eps_step

print("==> generating datasets with the following attack parameters:")
print(f"   * num_eps: {n_epsilons}")
print(f"   * min_eps: {min_eps}")
print(f"   * max_eps: {max_eps}")
print(f"   * eps_step: {eps_step}")

for idx_eps, eps in enumerate(epsilons):

    attacked_dataset = {}
    attacked_dataset["fgsm"] = []
    attacked_dataset["pgd"] = []
    # Define path to attacked dataset 
    path_eps = {} 
    base_path = Path("../data/cifar10_attacked")
    base_path.mkdir(parents=True, exist_ok=True)
    path_eps["fgsm"] = base_path / Path(f"cifar10_fgsm_eps_{eps:0.3f}.pt")
    path_eps["pgd"] = base_path / Path(f"cifar10_pgd_eps_{eps:0.3f}.pt")

    # Iterate over dataset
    for x_orig, label in test_loader:
        x_orig, label = x_orig.to(device), label.to(device)
        k = steps[idx_eps]
        # Generate adversarial attack sample for FGSM attack
        x_adv_fgsm = fgsm(model, x=x_orig, label=label, eps=eps, targeted=False, clip_min=0, clip_max=1).detach()
        # Generate adversarial attack sample for PGD attack
        x_adv_pgd = pgd(model, x=x_orig, label=label, k=k, eps=eps, eps_step=eps_step, targeted=False, clip_min=0, clip_max=1).detach()

        attacked_dataset["fgsm"].extend(zip(x_adv_fgsm.to(torch.device("cpu")), label[:,None]))
        attacked_dataset["pgd"].extend(zip(x_adv_pgd.to(torch.device("cpu")), label[:,None]))

    # Save attacked dataset
    for atk in ["fgsm", "pgd"]:
        print(attacked_dataset[atk][0][0].shape, attacked_dataset[atk][0][1].shape)
        torch.save(attacked_dataset[atk], path_eps[atk])
            
