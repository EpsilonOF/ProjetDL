import torch
import torchvision as tv
import numpy as np
import normflows as nf
import matplotlib.pyplot as plt
from tqdm import tqdm
import streamlit as st
import os


# Define the Glow model setup
def setup_model(
    num_classes=10,
    L=2,
    K=16,
    hidden_channels=256,
    split_mode="channel",
    scale=True,
    use_lu=True,
    dataset="MNIST",
):
    """
    Set up Glow model with multi-scale architecture

    Args:
        num_classes: Number of classes for class-conditional model
        L: Number of levels in the multi-scale architecture
        K: Number of flow steps per level
        hidden_channels: Number of channels in hidden layers
        split_mode: How to split variables ('channel' or 'checkerboard')
        scale: Whether to use scaling in affine coupling layers
        use_lu: Whether to use LU decomposition in 1x1 convolutions
        dataset: Dataset to use ("MNIST" or "CIFAR10")

    Returns:
        model: The Glow model
        num_classes: Number of classes
        n_dims: Dimensionality of the data
        device: Device to use (CUDA or CPU)
    """
    # Set seed for reproducibility
    torch.manual_seed(3008)

    # Configure input shape based on dataset
    if dataset == "MNIST":
        input_shape = (1, 28, 28)  # MNIST: grayscale, 28x28
        channels = 1
    elif dataset == "CIFAR10":
        input_shape = (3, 32, 32)  # CIFAR10: RGB, 32x32
        channels = 3
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    n_dims = np.prod(input_shape)

    # Configure flows, distributions, and merge operations
    q0 = []
    merges = []
    flows = []

    for i in range(L):
        flows_ = []
        # Add K flow steps of Glow blocks
        for j in range(K):
            flows_ += [
                nf.flows.GlowBlock(
                    channels * 2 ** (L + 1 - i),
                    hidden_channels,
                    split_mode=split_mode,
                    scale=scale,
                    use_lu=use_lu,
                )
            ]
        # Add squeeze operation at the end of each level
        flows_ += [nf.flows.Squeeze()]
        flows += [flows_]

        # Configure merges and base distributions
        if i > 0:
            merges += [nf.flows.Merge()]
            latent_shape = (
                input_shape[0] * 2 ** (L - i),
                input_shape[1] // 2 ** (L - i),
                input_shape[2] // 2 ** (L - i),
            )
        else:
            latent_shape = (
                input_shape[0] * 2 ** (L + 1),
                input_shape[1] // 2**L,
                input_shape[2] // 2**L,
            )
        q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes)]

    # Construct flow model with multi-scale architecture
    model = nf.MultiscaleFlow(q0, flows, merges)

    # Move model to GPU if available
    enable_cuda = True
    device = torch.device(
        "cuda" if torch.cuda.is_available() and enable_cuda else "cpu"
    )
    model = model.to(device)

    return model, num_classes, n_dims, device


def prepare_training_data(batch_size=128, dataset="MNIST"):
    """
    Prepare data loaders for training

    Args:
        batch_size: Batch size for training
        dataset: Dataset to use ("MNIST" or "CIFAR10")

    Returns:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        train_iter: Iterator for training data
    """
    # Common transformations
    if dataset == "MNIST":
        # MNIST transformations
        transform = tv.transforms.Compose(
            [
                tv.transforms.ToTensor(),
                nf.utils.Scale(255.0 / 256.0),
                nf.utils.Jitter(1 / 256.0),
            ]
        )

        # Load MNIST dataset
        train_data = tv.datasets.MNIST(
            "datasets/", train=True, download=True, transform=transform
        )
        test_data = tv.datasets.MNIST(
            "datasets/", train=False, download=True, transform=transform
        )

    elif dataset == "CIFAR10":
        # CIFAR10 transformations
        transform = tv.transforms.Compose(
            [
                tv.transforms.ToTensor(),
                nf.utils.Scale(255.0 / 256.0),
                nf.utils.Jitter(1 / 256.0),
            ]
        )

        # Load CIFAR10 dataset
        train_data = tv.datasets.CIFAR10(
            "datasets/", train=True, download=True, transform=transform
        )
        test_data = tv.datasets.CIFAR10(
            "datasets/", train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    train_iter = iter(train_loader)

    return train_loader, test_loader, train_iter


def train_model(
    model,
    train_loader,
    train_iter,
    device,
    max_iter=20000,
    save_interval=1000,
    model_path="data/glow_model.pth",
    show_progress=True,
):
    """
    Train the Glow model

    Args:
        model: The Glow model to train
        train_loader: DataLoader for training data
        train_iter: Iterator for training data
        device: Device to use (CUDA or CPU)
        max_iter: Maximum number of iterations
        save_interval: Interval for saving model checkpoints
        model_path: Path to save the model
        show_progress: Whether to show progress bars and plots

    Returns:
        loss_hist: History of loss values during training
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    loss_hist = np.array([])
    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3, weight_decay=1e-5)

    # Setup progress tracking
    iter_range = tqdm(range(max_iter)) if show_progress else range(max_iter)

    for i in iter_range:
        # Get batch of data
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        # Forward pass and loss calculation
        optimizer.zero_grad()
        loss = model.forward_kld(x.to(device), y.to(device))

        # Backward pass and optimization step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        # Record loss
        loss_hist = np.append(loss_hist, loss.detach().to("cpu").numpy())

        # Free memory
        del x, y, loss

        # Save checkpoints
        if (i + 1) % save_interval == 0:
            torch.save(model.state_dict(), model_path)
            if show_progress:
                print(f"Model saved at iteration {i + 1}")

    # Save final model
    torch.save(model.state_dict(), model_path)

    # Plot loss history if requested
    if show_progress:
        plt.figure(figsize=(10, 6))
        plt.plot(loss_hist, label="Loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training Loss History")
        plt.legend()
        plt.savefig("images/glow/training_loss.png")
        plt.close()

    return loss_hist


def get_bits_per_dim(model, test_loader, n_dims, device):
    """
    Calculate bits per dimension (compression metric) for the model

    Args:
        model: Trained Glow model
        test_loader: DataLoader for test data
        n_dims: Dimensionality of the data
        device: Device to use (CUDA or CPU)

    Returns:
        bpd: Bits per dimension value
    """
    n = 0
    bpd_cum = 0

    with torch.no_grad():
        for x, y in iter(test_loader):
            # Calculate negative log likelihood
            nll = model(x.to(device), y.to(device))
            nll_np = nll.cpu().numpy()

            # Convert to bits per dimension
            bpd_cum += np.nansum(nll_np / np.log(2) / n_dims + 8)
            n += len(x) - np.sum(np.isnan(nll_np))

    bpd = bpd_cum / n
    return bpd
