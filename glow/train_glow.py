# Import required packages
import torch
import torchvision as tv
import numpy as np
import normflows as nf
import matplotlib.pyplot as plt
from tqdm import tqdm


def create_glow_model(
    input_shape=(1, 28, 28),
    L=1,
    K=14,
    hidden_channels=256,
    split_mode="channel",
    scale=True,
    num_classes=10,
):
    """
    Create a multi-scale Glow model.

    This function builds a Glow model with specified architecture parameters.
    The model consists of multiple levels, each with a sequence of GlowBlocks.

    Args:
        input_shape (tuple, optional): Shape of the input data. Defaults to (1, 28, 28).
        L (int, optional): Number of levels in the multi-scale architecture. Defaults to 1.
        K (int, optional): Number of flow steps per level. Defaults to 14.
        hidden_channels (int, optional): Number of hidden channels in coupling layers.
        Defaults to 256.
        split_mode (str, optional): Mode for splitting tensors. Defaults to 'channel'.
        scale (bool, optional): Whether to use scaling in affine coupling layers.
        Defaults to True.
        num_classes (int, optional): Number of classes for conditional model.
        Defaults to 10.

    Returns:
        normflows.MultiscaleFlow: The configured Glow model
    """
    torch.manual_seed(0)  # For reproducibility

    channels = input_shape[0]

    q0 = []
    merges = []
    flows = []

    for i in range(L):
        flows_ = []
        for j in range(K):
            flows_ += [
                nf.flows.GlowBlock(
                    channels * 2 ** (L + 1 - i),
                    hidden_channels,
                    split_mode=split_mode,
                    scale=scale,
                )
            ]
        flows_ += [nf.flows.Squeeze()]
        flows += [flows_]

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

    model = nf.MultiscaleFlow(q0, flows, merges)
    return model


def prepare_data_loaders(batch_size=128):
    """
    Prepare MNIST data loaders for training and testing.

    This function creates PyTorch DataLoader objects for the MNIST dataset,
    applying appropriate transformations.

    Args:
        batch_size (int, optional): Batch size for data loading. Defaults to 128.

    Returns:
        tuple: A tuple containing:
            - train_loader (torch.utils.data.DataLoader): DataLoader for training data
    """
    transform = tv.transforms.Compose(
        [
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(
                (0.1307,), (0.3081,)
            ),  # Standard normalization for MNIST
        ]
    )

    # Load MNIST datasets
    train_data = tv.datasets.MNIST(
        "datasets/", train=True, download=True, transform=transform
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True
    )

    return train_loader


def train_model(
    model, train_loader, device, max_iter=20000, lr=1e-3, weight_decay=1e-5
):
    """
    Train the Glow model.

    This function trains a Glow model using stochastic gradient descent on MNIST data.
    It tracks the loss history for visualization.

    Args:
        model (normflows.MultiscaleFlow): The Glow model to train
        train_loader (torch.utils.data.DataLoader): DataLoader for training data
        device (torch.device): Device to use for training (CPU/GPU)
        max_iter (int, optional): Maximum number of training iterations. Defaults to 20000.
        lr (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
        weight_decay (float, optional): Weight decay for regularization. Defaults to 1e-5.

    Returns:
        tuple: A tuple containing:
            - model (normflows.MultiscaleFlow): The trained model
            - loss_hist (numpy.ndarray): History of training losses
    """
    loss_hist = np.array([])
    optimizer = torch.optim.Adamax(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_iter = iter(train_loader)

    for i in tqdm(range(max_iter)):
        # Get next batch, reinitialize data iterator when needed
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        # Training step
        optimizer.zero_grad()
        loss = model.forward_kld(x.to(device), y.to(device))

        # Skip problematic iterations (NaN or inf)
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        # Record loss history
        loss_hist = np.append(loss_hist, loss.detach().to("cpu").numpy())

        del (x, y, loss)  # Free up memory

    return model, loss_hist


def plot_loss_history(loss_hist):
    """
    Plot the training loss history.

    Args:
        loss_hist (numpy.ndarray): Array of loss values during training
    """
    plt.figure(figsize=(10, 10))
    plt.plot(loss_hist, label="loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Glow Training Loss")
    plt.legend()
    plt.show()


def generate_samples(model, device, num_classes=10, num_samples_per_class=10):
    """
    Generate and visualize samples from the trained model.

    This function generates a grid of samples from the trained Glow model,
    with each row corresponding to a different class.

    Args:
        model (normflows.MultiscaleFlow): The trained Glow model
        device (torch.device): Device to use for inference (CPU/GPU)
        num_classes (int, optional): Number of classes to sample. Defaults to 10.
        num_samples_per_class (int, optional): Number of samples per class. Defaults to 10.
    """
    with torch.no_grad():
        y = torch.arange(num_classes).repeat(num_samples_per_class).to(device)
        x, _ = model.sample(y=y)
        x_ = torch.clamp(x, 0, 1)

        plt.figure(figsize=(15, 15))
        plt.imshow(
            np.transpose(
                tv.utils.make_grid(x_, nrow=num_classes).cpu().numpy(),
                (1, 2, 0),
            )
        )
        plt.axis("off")
        plt.title("Generated Samples by Class")
        plt.show()


def save_model(model, path="model_glow.pth"):
    """
    Save the trained model to disk.

    Args:
        model (normflows.MultiscaleFlow): The trained model to save
        path (str, optional): Path where to save the model. Defaults to "model_glow.pth".
    """
    torch.save(model.state_dict(), path)
    print(f"Model saved to '{path}'")


def load_model(model, path="model_glow.pth", device=None):
    """
    Load a trained model from disk.

    Args:
        model (normflows.MultiscaleFlow): The model architecture to load weights into
        path (str, optional): Path from which to load the model. Defaults to "model_glow.pth".
        device (torch.device, optional): Device to load the model to. Defaults to None.

    Returns:
        normflows.MultiscaleFlow: The loaded model
    """
    if device is None:
        model.load_state_dict(torch.load(path))
    else:
        model.load_state_dict(torch.load(path, map_location=device))
    print(f"Model loaded from '{path}'")
    return model


def execute_glow(L=1, K=14, batch_size=128, max_iter=20000, save_path="modele.pth"):
    """
    Main function to train a Glow model on MNIST.

    This function coordinates the entire training pipeline:
    - Model creation
    - Data preparation
    - Training
    - Visualization
    - Model saving

    Args:
        L (int, optional): Number of levels in the multi-scale architecture. Defaults to 1.
        K (int, optional): Number of flow steps per level. Defaults to 14.
        batch_size (int, optional): Batch size for training. Defaults to 128.
        max_iter (int, optional): Maximum number of training iterations. Defaults to 20000.
        save_path (str, optional): Path where to save the model. Defaults to "modele.pth".
    """
    # Create model
    input_shape = (1, 28, 28)
    hidden_channels = 256
    model = create_glow_model(
        input_shape=input_shape, L=L, K=K, hidden_channels=hidden_channels
    )

    # Move model to GPU if available
    enable_cuda = True
    device = torch.device(
        "cuda" if torch.cuda.is_available() and enable_cuda else "cpu"
    )
    model = model.to(device)
    print(f"Using device: {device}")

    # Prepare data
    train_loader = prepare_data_loaders(batch_size=batch_size)

    # Train model
    print(f"Starting training for {max_iter} iterations...")
    model, loss_hist = train_model(
        model=model,
        train_loader=train_loader,
        device=device,
        max_iter=max_iter,
    )

    # Plot training loss
    plot_loss_history(loss_hist)

    # Save model
    save_model(model, path=save_path)

    # Generate samples
    print("Generating samples from trained model...")
    generate_samples(model, device)
