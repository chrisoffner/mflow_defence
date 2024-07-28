import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader
from utils.attacks import fgsm, pgd
from utils.utils import resnet_CIFAR10

from numpy.random import choice

import matplotlib.pyplot as plt

import os
import argparse
import tarfile
from pathlib import Path
import math

parser = argparse.ArgumentParser(description="Plot Resnet-50 Accuracies on attacked CIFAR-10 datasets")
parser.add_argument('--resnet', '-r', type=Path, help='path to trained resnet checkpoint', required=False, default="../models/resnet/resnet50_cifar10.pt")

parser.add_argument('--samples', '-s', type=int, help='number of image samples, maximum 10000', required=False, default=1_000)

parser.add_argument('--num_eps', '-n', type=int, help='number of epsilons', required=False, default=50)
parser.add_argument('--min_eps', '-e', type=float, help='minimum value of attack epsilon', required=False, default=0.0)
parser.add_argument('--max_eps', '-E', type=float, help='maximum value of attack epsilon', required=False, default=0.1)

parser.add_argument('--plot_only', '-p', action='store_true', help='plot only by reading results from standard path')

args = parser.parse_args()

# Set device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"Using device: {device}")

# define range of epsilons to try attacks with
n_epsilons = args.num_eps
min_eps = args.min_eps
max_eps = args.max_eps
epsilons = torch.linspace(min_eps, max_eps, n_epsilons)

print("==> generating figure with attacked datasets with the following attack parameters:")
print(f"   * num_eps: {n_epsilons}")
print(f"   * min_eps: {min_eps}")
print(f"   * max_eps: {n_epsilons}")

attacks = ["fgsm", "pgd"]

results = { atk: torch.zeros(n_epsilons, 2) for atk in attacks }

results_dir = Path("../data/experimental_results/attacked_undefended_resnet/")
results_dir.mkdir(parents=True, exist_ok=True)
path_results = { atk: results_dir / f"accs_{atk}.pt" for atk in attacks }

if not args.plot_only:
    # Define transforms
    transform = transforms.ToTensor()

    # Normalization is performed only right before passing the input to the model
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    # Load CIFAR-10 test dataset
    bs = 100
    subset_indices = choice(10_000, size=args.samples, replace=False)
    test_data = torch.utils.data.Subset(
        torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=True, transform=transform),
        subset_indices)
    test_loader = DataLoader(test_data, batch_size=bs, shuffle=False)
    print(f"original dataset size: {len(test_data)}")

    # Validate ResNet checkpoint path
    assert os.path.exists(args.resnet), 'Error: no checkpoint file found!'

    # Init modified ResNet-50
    classifier = resnet_CIFAR10(args.resnet, device)
    classifier.to(device)
    classifier.eval()

    # Iterate over epsilons
    for idx_eps, eps in enumerate(epsilons):
        print(f"eps: {eps:0.3f}")
        attacked_data = {}
        attacked_dataloaders = {}
        iterators = {}
        for atk in attacks:
            file_path = Path(f"../data/cifar10_attacked/cifar10_{atk}_eps_{eps:0.3f}.tar.gz")
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=file_path.parent)
            attacked_data[atk] = torch.utils.data.Subset(
                torch.load(str(file_path).replace("tar.gz", "pt"), map_location=device),
                subset_indices)
            os.remove(str(file_path).replace("tar.gz", "pt"))
            attacked_dataloaders[atk] = DataLoader(attacked_data[atk], batch_size=bs, shuffle=False)
            iterators[atk] = iter(attacked_dataloaders[atk])
            print(f"attacked ({atk}, eps={eps:0.3f}) dataset size: {len(attacked_data[atk])}")

        # Iterate over dataset
        for _, labels in test_loader:
            # Generate adversarial attack sample for attacks
            x_adv = {}
            for atk in attacks:
                x_adv[atk], _ = next(iterators[atk])

            # Perform inference
            with torch.no_grad():
                pred_adv = {}
                for atk in attacks:
                    x_adv_normalized = normalize(x_adv[atk]).to(device)
                    outputs = classifier(x_adv_normalized)
                    _, pred_adv[atk] = torch.max(outputs, 1)

            for atk in attacks:
                pred_success = (pred_adv[atk] == labels)
                # correct classifications
                results[atk][idx_eps, 0] += pred_success.sum().item()
                results[atk][idx_eps, 1] += labels.size(0) - pred_success.sum().item()

        if idx_eps % 10 == 0:
            for atk in attacks:
                pth = path_results[atk]
                torch.save(results[atk], pth.parent / (pth.stem + f"_ckpt_{idx_eps}_eps_{eps:0.3f}" + pth.suffix))

figures_dir = Path("../figures/attacked_undefended_resnet/")
figures_dir.mkdir(parents=True, exist_ok=True)

for atk in attacks:
    if args.plot_only:
        results[atk] = torch.load(path_results[atk], map_location=device)
    else: 
        torch.save(results[atk], path_results[atk])
    # Normalize all cases so we get probabilities that sum to 1
    results[atk] /= results[atk].sum(dim=1)[:,None]
    correct, _ = results[atk].T
    plt.figure(figsize=(5, 4), dpi=200)
    plt.plot(epsilons, correct)

    plt.suptitle(f"Accuracy of ResNet-50 vs {atk.upper()} Attack (CIFAR-10)")
    plt.title(f"max acc: {correct.max():.2%}, min acc: {correct.min():.2%}")
    plt.xlabel(r"Perturbation Magnitude ($\epsilon$)")
    plt.ylabel("Classification Accuracy")
    plt.grid(color='lightgrey', linestyle='-')
    plt.gca().set_ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: '{:.3g}'.format(x)))
    plt.margins(0)

    plt.tight_layout()
    plt.savefig(figures_dir / f"accuracy_undefended_resnet_vs_{atk}.pdf")