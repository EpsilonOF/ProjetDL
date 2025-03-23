import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import gaussian_filter
import os
import torchvision as tv
import torchvision.datasets as datasets
import streamlit as st


class StandardLogisticDistribution:
    """
    A standard logistic distribution for use as a prior in normalizing flows.

    This class implements a logistic distribution with location parameter 0
    and scale parameter 1, providing methods for computing log probability
    and sampling from the distribution.

    Attributes:
        dim (int): Dimensionality of the distribution
        device (str): Device for tensor operations ('cpu' or 'cuda')
    """

    def __init__(self, dim=2, device="cpu"):
        """
        Initialize the standard logistic distribution.

        Args:
            dim (int): Dimensionality of the distribution
            device (str): Device for tensor operations ('cpu' or 'cuda')
        """
        self.dim = dim
        self.device = device

    def log_prob(self, z):
        """
        Compute the log probability of samples under the standard logistic distribution.

        Args:
            z (torch.Tensor): Input tensor of shape [batch_size, dim]

        Returns:
            torch.Tensor: Log probability for each input sample of shape [batch_size]
        """
        return -torch.sum(z + 2 * torch.nn.functional.softplus(-z), dim=1)

    def sample(self, num_samples):
        """
        Generate samples from the standard logistic distribution.

        Args:
            num_samples (int): Number of samples to generate

        Returns:
            torch.Tensor: Samples from the logistic distribution of shape [num_samples, dim]
        """
        u = torch.rand(num_samples, self.dim, device=self.device)
        return torch.log(u) - torch.log(1 - u)


class CouplingLayer(nn.Module):
    """
    Additive coupling layer for NICE model.

    This layer implements the additive coupling transform where part of the input
    is kept fixed and used to parameterize a translation of the remaining part.
    The layer is invertible and has a Jacobian determinant of 1.

    Attributes:
        input_dim (int): Dimensionality of the input
        hidden_dim (int): Dimensionality of the hidden layers in the coupling network
        mask (torch.Tensor): Binary mask determining which dimensions remain fixed
        swap (bool): Whether to swap the roles of masked/unmasked variables after transformation
        net (nn.Sequential): Neural network that parameterizes the coupling transform
    """

    def __init__(self, input_dim, hidden_dim=128, mask=None, swap=False):
        """
        Initialize the coupling layer.

        Args:
            input_dim (int): Dimensionality of the input
            hidden_dim (int): Dimensionality of the hidden layers in the coupling network
            mask (torch.Tensor, optional): Binary mask determining which dimensions remain fixed.
                If None, the first half of dimensions are masked.
            swap (bool, optional): Whether to swap the roles of masked/unmasked variables
                after transformation. Defaults to False.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.swap = swap

        if mask is None:
            # Default mask splits input in half
            self.mask = torch.zeros(input_dim)
            self.mask[: input_dim // 2] = 1
        else:
            self.mask = mask

        # Create the coupling network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

        # Initialize weights with small random values
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.05)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Forward transformation from data space to latent space.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]

        Returns:
            torch.Tensor: Transformed tensor of shape [batch_size, input_dim]
        """
        # Split input based on mask
        x_masked = x * self.mask

        # Compute coupling transform
        translation = self.net(x_masked) * (1 - self.mask)

        # Apply coupling transform
        y = x + translation

        if self.swap:
            # Swap the masked and unmasked variables (permutation)
            mask_swap = 1 - self.mask
            y = x_masked + (y * mask_swap)

        return y

    def inverse(self, y):
        """
        Inverse transformation from latent space to data space.

        Args:
            y (torch.Tensor): Input tensor of shape [batch_size, input_dim]

        Returns:
            torch.Tensor: Transformed tensor of shape [batch_size, input_dim]
        """
        # Split input based on mask
        y_masked = y * self.mask

        # Compute coupling transform
        translation = self.net(y_masked) * (1 - self.mask)

        # Apply inverse coupling transform
        x = y - translation

        if self.swap:
            # Swap the masked and unmasked variables (permutation)
            mask_swap = 1 - self.mask
            x = y_masked + (x * mask_swap)

        return x


class NICE(nn.Module):
    """
    Implementation of NICE (Non-linear Independent Components Estimation) model.

    NICE is a normalizing flow model that transforms a simple prior distribution
    into a complex target distribution using a series of invertible transformations.
    It consists of multiple additive coupling layers and a scaling layer.

    Attributes:
        input_dim (int): Dimensionality of the input data
        hidden_dim (int): Dimensionality of the hidden layers in coupling networks
        num_layers (int): Number of coupling layers
        prior (StandardLogisticDistribution): Prior distribution
        coupling_layers (nn.ModuleList): List of coupling layers
        scaling (nn.Parameter): Learnable scaling parameters
    """

    def __init__(self, input_dim, hidden_dim=128, num_layers=4):
        """
        Initialize the NICE model.

        Args:
            input_dim (int): Dimensionality of the input data
            hidden_dim (int): Dimensionality of the hidden layers in coupling networks
            num_layers (int): Number of coupling layers
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Prior distribution
        self.prior = StandardLogisticDistribution(input_dim)

        # Create coupling layers
        self.coupling_layers = nn.ModuleList()
        for i in range(num_layers):
            # Create alternating masks
            if i % 2 == 0:
                mask = torch.zeros(input_dim)
                mask[: input_dim // 2] = 1
            else:
                mask = torch.zeros(input_dim)
                mask[input_dim // 2 :] = 1

            self.coupling_layers.append(CouplingLayer(input_dim, hidden_dim, mask))

        # Scaling parameters (initialized to zeros)
        self.scaling = nn.Parameter(torch.zeros(input_dim))

    def forward(self, x):
        """
        Transform from data space to latent space.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]

        Returns:
            torch.Tensor: Transformed tensor in latent space
        """
        z = x

        # Apply coupling layers
        for layer in self.coupling_layers:
            z = layer(z)

        # Apply scaling
        z = z * torch.exp(self.scaling)

        return z

    def inverse(self, z):
        """
        Transform from latent space to data space.

        Args:
            z (torch.Tensor): Input tensor in latent space

        Returns:
            torch.Tensor: Transformed tensor in data space
        """
        x = z

        # Apply inverse scaling
        x = x * torch.exp(-self.scaling)

        # Apply inverse coupling layers in reverse order
        for layer in reversed(self.coupling_layers):
            x = layer.inverse(x)

        return x

    def log_prob(self, x):
        """
        Compute log probability of samples under the model.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]

        Returns:
            torch.Tensor: Log probability for each input sample
        """
        z = self.forward(x)
        log_det_jacobian = torch.sum(self.scaling)
        log_prior = self.prior.log_prob(z)

        return log_prior + log_det_jacobian

    def sample(self, num_samples):
        """
        Generate samples from the model.

        Args:
            num_samples (int): Number of samples to generate

        Returns:
            torch.Tensor: Generated samples of shape [num_samples, input_dim]
        """
        z = self.prior.sample(num_samples)
        x = self.inverse(z)

        return x

    def save(self, path):
        """
        Save the model to a file.

        Args:
            path (str): Path to save the model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
        Load the model from a file.

        Args:
            path (str): Path to the saved model
        """
        self.load_state_dict(torch.load(path))


class ImageDistribution:
    def __init__(self, image_path, resolution=200, blur_sigma=1.0, scale=6.0):
        super().__init__()
        try:
            # Print to verify image loading
            print(f"Loading image: {image_path}")
            img = Image.open(image_path).convert("L")
            print(f"Image loaded, dimensions: {img.size}")

            # Resize and process image
            img = img.resize((resolution, resolution))
            self.prob = np.array(img).astype(np.float32)

            # Invert if needed (for images with white background)
            if self.prob.mean() > 127:
                self.prob = 255 - self.prob

            # Normalize
            self.prob = self.prob / (self.prob.sum() + 1e-10)

            # Smooth
            self.prob = gaussian_filter(self.prob, sigma=blur_sigma)
            self.prob = self.prob / (self.prob.sum() + 1e-10)

            # Create coordinate grid
            self.resolution = resolution
            self.scale = scale
            x = np.linspace(-scale / 2, scale / 2, resolution)
            y = np.linspace(-scale / 2, scale / 2, resolution)
            self.xx, self.yy = np.meshgrid(x, y)

            # Convert to tensor
            self.prob_tensor = torch.tensor(self.prob, dtype=torch.float32)

            # Verify distribution validity
            print(f"Sum of probabilities: {self.prob.sum()}")
            print(f"Min/Max probabilities: {self.prob.min()}/{self.prob.max()}")

        except Exception as e:
            print(f"Error loading image: {e}")
            raise

    def log_prob(self, z):
        z_np = z.detach().cpu().numpy()
        x_idx = np.clip(
            ((z_np[:, 0] + self.scale / 2) / self.scale * self.resolution).astype(int),
            0,
            self.resolution - 1,
        )
        y_idx = np.clip(
            ((z_np[:, 1] + self.scale / 2) / self.scale * self.resolution).astype(int),
            0,
            self.resolution - 1,
        )

        probs = self.prob[y_idx, x_idx]
        log_probs = np.log(probs + 1e-10)

        return torch.tensor(log_probs, dtype=torch.float32, device=z.device)

    def sample(self, num_samples):
        # Create flattened probability distribution for sampling
        flat_prob = self.prob.flatten()
        indices = np.random.choice(len(flat_prob), size=num_samples, p=flat_prob)

        # Convert indices to 2D coordinates
        y_idx = indices // self.resolution
        x_idx = indices % self.resolution

        # Convert to continuous space
        x = (x_idx / self.resolution - 0.5) * self.scale
        y = (y_idx / self.resolution - 0.5) * self.scale

        samples = np.column_stack([x, y])
        return torch.tensor(samples, dtype=torch.float32)


class MNISTDistribution:
    def __init__(self, batch_size=128, download=True):
        """
        Creates a distribution based on MNIST dataset

        Args:
            batch_size: Batch size for loading data
            download: Whether to download the dataset if not present
        """
        self.batch_size = batch_size

        # Define transform to normalize the data
        transform = tv.transforms.Compose(
            [
                tv.transforms.ToTensor(),
                tv.transforms.Lambda(
                    lambda x: (x.view(-1) * 2) - 1
                ),  # Scale to [-1, 1] and flatten
            ]
        )

        # Download and load the training data
        self.mnist_trainset = datasets.MNIST(
            root="./datasets",
            train=True,
            download=download,
            transform=transform,
        )

        # Create data loader
        self.train_loader = torch.utils.data.DataLoader(
            self.mnist_trainset, batch_size=batch_size, shuffle=True
        )

        self.iter = iter(self.train_loader)

        # MNIST dimensions
        self.dim = 28 * 28

    def log_prob(self, z):
        """
        Approximate log probability of MNIST digits.
        This is a simplification, not actual log prob.
        """
        # This is a very simplistic approximation - not an actual density
        z_norm = torch.norm(z, dim=1)
        return -z_norm / 10.0

    def sample(self, num_samples):
        """Sample from MNIST dataset."""
        # Get a new batch if needed
        try:
            data, _ = next(self.iter)
        except StopIteration:
            self.iter = iter(self.train_loader)
            data, _ = next(self.iter)

        # Get the requested number of samples
        if num_samples <= data.size(0):
            samples = data[:num_samples]
        else:
            # If more samples requested than batch size, repeat the batch
            repeats = num_samples // data.size(0) + 1
            samples = data.repeat(repeats, 1)[:num_samples]

        return samples


def setup_model(num_layers=4, hidden_dim=128):
    """
    Set up NICE model and target distribution.

    This function configures a NICE model with the specified parameters and
    initializes an appropriate target distribution.

    Args:
        num_layers (int, optional): Number of coupling layers. Defaults to 4.
        hidden_dim (int, optional): Hidden dimensions in coupling networks. Defaults to 128.

    Returns:
        tuple: A tuple containing:
            - model (NICE): The configured NICE model
            - device (torch.device): Device for tensor operations (CPU/GPU)
            - target: Target distribution object (TwoMoons, ImageDistribution, or MNISTDistribution)
    """
    # Set up device
    enable_cuda = True
    device = torch.device(
        "cuda" if torch.cuda.is_available() and enable_cuda else "cpu"
    )

    # Create model based on dataset type
    input_dim = 2
    model = NICE(input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)

    # Create target distribution
    target = TwoMoons()

    return model, device, target


class TwoMoons:
    """
    Two moons distribution for 2D density estimation.

    This class implements the two moons distribution commonly used as a benchmark
    for density estimation algorithms. It provides methods for sampling from the
    distribution and computing log probabilities.

    Attributes:
        n_dims (int): Dimensionality of the distribution (fixed at 2)
    """

    def __init__(self):
        """
        Initialize the two moons distribution.
        """
        self.n_dims = 2

    def sample(self, num_samples):
        """
        Sample from the two moons distribution using rejection sampling.

        This implementation is similar to the normflows library implementation.

        Args:
            num_samples (int): Number of samples to generate

        Returns:
            torch.Tensor: Samples from the two moons distribution of shape [num_samples, 2]
        """
        # Using direct sampling with rejection
        samples = []
        count = 0

        # Use rejection sampling for two moons
        while count < num_samples:
            # Sample uniformly from a rectangle
            x = torch.rand(min(10 * num_samples, 1000), 2) * 8 - 4  # Range from -4 to 4

            # Calculate the log probability
            log_prob = self.log_prob(x)

            # Convert to probability and normalize
            prob = torch.exp(log_prob - torch.max(log_prob))
            prob = prob / torch.max(prob)

            # Rejection sampling
            mask = torch.rand(x.shape[0]) < prob
            x_accepted = x[mask]

            # Add accepted samples
            samples.append(x_accepted)
            count += x_accepted.shape[0]

            # Break if we have enough samples
            if count >= num_samples:
                break

        # Combine and truncate
        samples = torch.cat(samples, dim=0)[:num_samples]
        return samples

    def log_prob(self, z):
        """
        Compute the log probability of samples under the two moons distribution.

        This implementation matches the normflows library implementation.

        Args:
            z (torch.Tensor): Input tensor of shape [batch_size, 2]

        Returns:
            torch.Tensor: Log probability for each input sample of shape [batch_size]
        """
        a = torch.abs(z[:, 0])
        log_prob = (
            -0.5 * ((torch.norm(z, dim=1) - 2) / 0.2) ** 2  # noqa: W503
            - 0.5 * ((a - 2) / 0.3) ** 2  # noqa: W503
            + torch.log(1 + torch.exp(-4 * a / 0.09))  # noqa: W503
        )
        return log_prob


def target_distribution(target, device, dataset_type="2d"):
    """
    Create a grid for visualizing the target distribution.

    This function creates a 2D grid for visualizing a probability distribution.
    For 2D distributions, it returns tensors that can be used for creating contour plots
    or heatmaps. For higher-dimensional distributions, it returns None values.

    Args:
        target: Target distribution object with a log_prob method
        device (torch.device): Device for tensor operations (CPU/GPU)
        dataset_type (str, optional): Type of dataset ("2d" or other). Defaults to "2d".

    Returns:
        tuple: For 2D distributions, a tuple containing:
            - xx (torch.Tensor): X-coordinates of the grid
            - yy (torch.Tensor): Y-coordinates of the grid
            - zz (torch.Tensor): Flattened grid points for evaluation
        For high-dimensional distributions, returns (None, None, None).
    """
    if dataset_type != "2d":
        return None, None, None

    grid_size = 200
    xx, yy = torch.meshgrid(
        torch.linspace(-4, 4, grid_size), torch.linspace(-4, 4, grid_size)
    )
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    zz = zz.to(device)

    log_prob = target.log_prob(zz).to("cpu").view(*xx.shape)
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    return xx, yy, zz


def train_model(
    model,
    target,
    device,
    xx=None,
    yy=None,
    zz=None,
    max_iter=4000,
    num_samples=512,
    show_iter=500,
    save_path=None,
    show_progress=True,
):
    """
    Train the NICE model on the target distribution.

    This function trains a NICE model using stochastic gradient descent on samples
    from the target distribution. It can also visualize the learning progress for
    2D distributions or MNIST.

    Args:
        model (NICE): NICE model to train
        target: Target distribution object with sample method
        device (torch.device): Device for tensor operations (CPU/GPU)
        xx (torch.Tensor, optional): X-coordinates grid for visualization.
        yy (torch.Tensor, optional): Y-coordinates grid for visualization.
        zz (torch.Tensor, optional): Flattened grid points for evaluation.
        max_iter (int, optional): Maximum number of training iterations. Defaults to 4000.
        num_samples (int, optional): Number of samples per training batch. Defaults to 512.
        show_iter (int, optional): Interval for showing visualization updates. Defaults to 500.
        save_path (str, optional): Path to save the trained model. If None, model is not saved.
        show_progress (bool, optional): Whether to show progress plots. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - model (NICE): The trained NICE model
            - loss_hist (numpy.ndarray): History of loss values during training
    """
    loss_hist = np.array([])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for it in tqdm(range(max_iter)):
        optimizer.zero_grad()

        # Get training samples
        x = target.sample(num_samples).to(device)

        # Calculate loss (negative log likelihood)
        loss = -model.log_prob(x).mean()

        # Backpropagation and optimization step
        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        # Record loss
        loss_hist = np.append(loss_hist, loss.to("cpu").data.numpy())

        # Display progress
        if show_progress and (it + 1) % show_iter == 0:
            print(f"Iteration {it + 1}, loss: {loss.item():.4f}")

        # Visualize 2D distributions
        model.eval()
        with torch.no_grad():
            samples = zz.view(-1, 2)
            log_prob = model.log_prob(samples)
        model.train()

        prob = torch.exp(log_prob.to("cpu").view(*xx.shape))
        prob[torch.isnan(prob)] = 0

        plt.figure(figsize=(10, 10))
        plt.pcolormesh(xx, yy, prob.data.numpy(), cmap="coolwarm")
        plt.gca().set_aspect("equal", "box")
        plt.title(f"Learned distribution - Iteration {it + 1}")
        plt.show()

    # Save model if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"Model saved to {save_path}")

    # Plot final loss history if requested
    if show_progress:
        plt.figure(figsize=(10, 5))
        plt.plot(loss_hist, label="loss")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Loss History")
        plt.legend()
        plt.show()

    return model, loss_hist


def plot_target_distribution(model, target, xx, yy, zz):
    """
    Plot 2D target and learned distributions side by side.

    This function creates a visualization comparing the target distribution
    with the distribution learned by the model, displayed side by side.

    Args:
        model (NICE): Trained NICE model
        target: Target distribution object with log_prob method
        xx (torch.Tensor): X-coordinates of the grid
        yy (torch.Tensor): Y-coordinates of the grid
        zz (torch.Tensor): Flattened grid points for evaluation
    """
    f, ax = plt.subplots(1, 2, sharey=True, figsize=(15, 7))

    # Plot target distribution
    log_prob = target.log_prob(zz).to("cpu").view(*xx.shape)
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    ax[0].pcolormesh(xx, yy, prob.data.numpy(), cmap="coolwarm")
    ax[0].set_aspect("equal", "box")
    ax[0].set_axis_off()
    ax[0].set_title("Target", fontsize=24)

    # Plot learned distribution
    model.eval()
    with torch.no_grad():
        log_prob = model.log_prob(zz.view(-1, 2)).to("cpu").view(*xx.shape)
    model.train()

    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    ax[1].pcolormesh(xx, yy, prob.data.numpy(), cmap="coolwarm")
    ax[1].set_aspect("equal", "box")
    ax[1].set_axis_off()
    ax[1].set_title("NICE", fontsize=24)

    plt.subplots_adjust(wspace=0.1)
    plt.show()


def execute_nice(n_layers, hidden_dim, max_iter):
    """
    Executes and trains a NICE (Non-linear Independent Components Estimation)
    model to learn a target distribution.

    This function configures and trains a NICE model with a specified number
    of layers and hidden dimensions, over a maximum number of iterations.
    It displays intermediate results and the final loss curve.

    Args:
        n_layers (int): The number of layers in the model.
        hidden_dim (int): The dimension of the hidden layers in the model.
        max_iter (int): The maximum number of iterations for training the model.

    Returns:
        None

    Description:
        - Configures the model parameters based on the specified number
        of layers and hidden dimension.
        - Initializes the NICE model with defined hyperparameters.
        - Trains the model over a specified number of iterations.
        - Displays thumbnails of the learned distribution at regular intervals.
        - Displays the loss curve at the end of training.

    Notes:
        - Uses the Adam optimizer with a learning rate of 1e-3 and weight decay of 1e-5.
        - Displays thumbnails of the learned distribution every `max_iter/10` iterations.
        - Handles NaN or infinite values in the loss to prevent training interruptions.
    """
    # Configuration et entraînement du modèle
    model, device, target = setup_model(num_layers=n_layers, hidden_dim=hidden_dim)
    xx, yy, zz = target_distribution(target, device)

    loss_hist = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    cols = st.columns(int((max_iter // (max_iter // 10)) / 2))
    cols2 = cols
    for it in range(max_iter):
        optimizer.zero_grad()
        x = target.sample(512).to(device)
        loss = -model.log_prob(x).mean()

        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        loss_hist.append(loss.item())

        if (it + 1) % (max_iter // 10) == 0:
            model.eval()
            with torch.no_grad():
                log_prob = model.log_prob(zz.view(-1, 2)).to("cpu").view(*xx.shape)
            prob = torch.exp(log_prob)
            prob[torch.isnan(prob)] = 0

            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pcolormesh(xx, yy, prob.numpy(), cmap="coolwarm")
            ax.set_aspect("equal", "box")
            ax.axis("off")
            ax.set_title(f"Itération {it + 1}", fontsize=10)
            if it // (max_iter // 10) >= len(cols):
                cols2[(it // (max_iter // 10)) % len(cols)].pyplot(fig)
            else:
                cols[it // (max_iter // 10)].pyplot(fig)
            plt.close(fig)  # Libération mémoire

            model.train()
    col1, col2 = st.columns(2)

    with col1:
        # Affichage final de la courbe de perte
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(loss_hist, label="Perte")
        ax_loss.set_xlabel("Itération")
        ax_loss.set_ylabel("Perte")
        ax_loss.set_title("Historique de la perte")
        ax_loss.legend()
        st.pyplot(fig_loss)
