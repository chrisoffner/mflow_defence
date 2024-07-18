import os
import argparse
from pathlib import Path
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

def load_cifar10(transform, test_batch_size, train_batch_size):
    dataloaders = {}
    dataset_sizes = {}

    # Load CIFAR-10 test dataset
    test_data = torchvision.datasets.CIFAR10(root="../data/cifar10", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=test_bs, shuffle=False, num_workers=2)

    dataloaders["val"] = test_loader
    dataset_sizes["val"] = len(test_data)

    # Load CIFAR-10 train dataset
    train_data = torchvision.datasets.CIFAR10(root="../data/cifar10", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=train_bs, num_workers=2, shuffle=True)

    dataloaders["train"] = train_loader
    dataset_sizes["train"] = len(train_data)

    return dataloaders, dataset_sizes

def train(model, trainloader, criterion, optimizer, device):
    train_loss = 0.0
    train_total = 0
    train_correct = 0

    # Switch to train mode
    model.train()

    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update training loss
        train_loss += loss.item() * inputs.size(0)

        # Compute training accuracy
        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # Compute average training loss and accuracy
    train_loss = train_loss / len(trainloader.dataset)
    train_accuracy = 100.0 * train_correct / train_total

    return model, train_loss, train_accuracy

def test(model, testloader, criterion, device):
    test_loss = 0.0
    test_total = 0
    test_correct = 0

    # Switch to evaluation mode
    model.eval()

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update test loss
            test_loss += loss.item() * inputs.size(0)

            # Compute test accuracy
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    # Compute average test loss and accuracy
    test_loss = test_loss / len(testloader.dataset)
    test_accuracy = 100.0 * test_correct / test_total

    return test_loss, test_accuracy

def train_epochs(model, trainloader, testloader, criterion, optimizer, scheduler, device, start_epoch, num_epochs, save_interval=5):
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print(f'Epoch {epoch + 1}/{start_epoch + num_epochs + 1}')
        print("training")
        model, train_loss, train_accuracy = train(model, trainloader, criterion, optimizer, device)
        test_loss, test_accuracy = test(model, testloader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(f'Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}%')
        print(f'Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%')
        print()

        scheduler.step()

        if (epoch + 1) % save_interval == 0:
            # Save the model and variables
            torch.save(model.state_dict(), f'resnet50_cifar10_{epoch+1}.pt')
            checkpoint = {
                'epoch': epoch + 1,
                'train_losses': train_losses,
                'train_accuracies': train_accuracies,
                'test_losses': test_losses,
                'test_accuracies': test_accuracies,
                'classes': classes
            }
            torch.save(checkpoint, f'resnet50_cifar10_variables_{epoch+1}.pt')

    return model, train_losses, train_accuracies, test_losses, test_accuracies

def plot_loss(train_losses, test_losses):
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(test_losses)), test_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    # plt.show()

def plot_accuracy(train_accuracies, test_accuracies):
    plt.figure()
    plt.plot(range(len(train_accuracies)), train_accuracies, label='Training Accuracy')
    plt.plot(range(len(test_accuracies)), test_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    # plt.show()

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
    parser.add_argument('--checkpoint', '-c', type=Path, help='path to checkpoint')
    parser.add_argument('--checkpoint_variables', '-V', type=Path, help='path to checkpoint variables')
    args = parser.parse_args()

    # Set device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load CIFAR-10 dataset
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    test_bs = 64
    train_bs = 64
    dataloaders, dataset_sizes = load_cifar10(transform=transform, test_batch_size=test_bs, train_batch_size=train_bs)
    print(f"dataloaders:\n{dataloaders}")
    print(f"dataset_sizes:\n{dataset_sizes}")
    # Define CIFAR-10 classes
    classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

    ## Load pre-trained ResNet-50 model
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    ## Freeze params
    #for param in model.parameters():
    #    param.requires_grad = False
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    start_epoch = 0
    print(f"checkpoint: {args.checkpoint}")
    print(f"variables: {args.checkpoint_variables}")
    if args.checkpoint is not None:
        assert os.path.exists(args.checkpoint), 'Error: no checkpoint file found!'
        print("==> loading from checkpoint")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint)  
        if args.checkpoint_variables is not None:
            assert os.path.exists(args.checkpoint_variables), 'Error: no checkpoint variables file found!'
            checkpoint_vars = torch.load(args.checkpoint_variables, map_location=device)
            start_epoch = checkpoint_vars["epoch"] - 1
    print(f"==> starting from epoch {start_epoch + 1}")
    # Modify the ResNet-50 architecture for CIFAR-10
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    model.to(device)
    model.eval()

    # train
    num_epochs = 30
    save_interval = 5
    model, train_losses, train_accuracies, test_losses, test_accuracies = train_epochs(model, dataloaders["train"], dataloaders["val"], criterion, optimizer, scheduler, device, start_epoch, num_epochs, save_interval)
    # Save the final trained model
    torch.save(model.state_dict(), f'resnet50_cifar10_final_model_epochs_{start_epoch + num_epochs + 1}.pt')
    torch.save({
        'epoch': start_epoch + num_epochs + 1,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'classes': classes
    }, f'resnet50_cifar10_final_variables_{start_epoch + num_epochs + 1}.pt')
    # Plot and save the loss and accuracy plots
    plot_loss(train_losses, test_losses)
    plot_accuracy(train_accuracies, test_accuracies)
