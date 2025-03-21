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

class StandardLogisticDistribution:
    def __init__(self, dim=2, device='cpu'):
        self.dim = dim
        self.device = device
        
    def log_prob(self, z):
        """Log probability of standard logistic distribution."""
        return -torch.sum(z + 2 * torch.nn.functional.softplus(-z), dim=1)
    
    def sample(self, num_samples):
        """Sample from standard logistic distribution."""
        u = torch.rand(num_samples, self.dim, device=self.device)
        return torch.log(u) - torch.log(1 - u)

class CouplingLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, mask=None, swap=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.swap = swap
        
        if mask is None:
            # Default mask splits input in half
            self.mask = torch.zeros(input_dim)
            self.mask[:input_dim//2] = 1
        else:
            self.mask = mask
            
        # Create the coupling network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
        # Initialize weights with small random values
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.05)
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
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
    def __init__(self, input_dim, hidden_dim=128, num_layers=4):
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
                mask[:input_dim//2] = 1
            else:
                mask = torch.zeros(input_dim)
                mask[input_dim//2:] = 1
                
            self.coupling_layers.append(CouplingLayer(input_dim, hidden_dim, mask))
            
        # Scaling parameters (initialized to zeros)
        self.scaling = nn.Parameter(torch.zeros(input_dim))
        
    def forward(self, x):
        """Transform from data space to latent space."""
        z = x
        
        # Apply coupling layers
        for layer in self.coupling_layers:
            z = layer(z)
            
        # Apply scaling
        z = z * torch.exp(self.scaling)
        
        return z
    
    def inverse(self, z):
        """Transform from latent space to data space."""
        x = z
        
        # Apply inverse scaling
        x = x * torch.exp(-self.scaling)
        
        # Apply inverse coupling layers in reverse order
        for layer in reversed(self.coupling_layers):
            x = layer.inverse(x)
            
        return x
    
    def log_prob(self, x):
        """Compute log probability of a sample."""
        z = self.forward(x)
        log_det_jacobian = torch.sum(self.scaling)
        log_prior = self.prior.log_prob(z)
        
        return log_prior + log_det_jacobian
    
    def sample(self, num_samples):
        """Generate samples from the model."""
        z = self.prior.sample(num_samples)
        x = self.inverse(z)
        
        return x
    
    def save(self, path):
        """Save the model to a file."""
        torch.save(self.state_dict(), path)
        
    def load(self, path):
        """Load the model from a file."""
        self.load_state_dict(torch.load(path))

class ImageDistribution:
    def __init__(self, image_path, resolution=200, blur_sigma=1.0, scale=6.0):
        super().__init__()
        try:
            # Print to verify image loading
            print(f"Loading image: {image_path}")
            img = Image.open(image_path).convert('L')
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
            x = np.linspace(-scale/2, scale/2, resolution)
            y = np.linspace(-scale/2, scale/2, resolution)
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
        x_idx = np.clip(((z_np[:, 0] + self.scale/2) / self.scale * self.resolution).astype(int),
                        0, self.resolution - 1)
        y_idx = np.clip(((z_np[:, 1] + self.scale/2) / self.scale * self.resolution).astype(int),
                        0, self.resolution - 1)
        
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
        transform = tv.transforms.Compose([
            tv.transforms.ToTensor(),
            tv.transforms.Lambda(lambda x: (x.view(-1) * 2) - 1)  # Scale to [-1, 1] and flatten
        ])
        
        # Download and load the training data
        self.mnist_trainset = datasets.MNIST(
            root='./datasets', 
            train=True,
            download=download, 
            transform=transform
        )
        
        # Create data loader
        self.train_loader = torch.utils.data.DataLoader(
            self.mnist_trainset,
            batch_size=batch_size,
            shuffle=True
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

def setup_model(image_path=None, num_layers=4, hidden_dim=128, dataset_type="2d"):
    """
    Set up NICE model and target distribution
    
    Args:
        image_path: Path to image to use as distribution (for 2D case)
        num_layers: Number of coupling layers
        hidden_dim: Hidden dimensions in coupling networks
        dataset_type: Type of dataset: "2d", "mnist", or "image"
    
    Returns:
        model: NICE model
        device: Device (CPU/GPU)
        target: Target distribution
    """
    # Set up device
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    
    # Create model based on dataset type
    if dataset_type == "mnist":
        input_dim = 28 * 28  # MNIST dimensions
        model = NICE(input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
        target = MNISTDistribution()
    else:  # 2D distributions
        input_dim = 2
        model = NICE(input_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
        
        # Create target distribution
        if dataset_type == "image" and image_path:
            print(f"Using image {image_path} as target distribution")
            target = ImageDistribution(image_path)
        else:
            print("Using TwoMoons distribution by default")
            target = TwoMoons()
        
    return model, device, target

class TwoMoons:
    def __init__(self):
        self.n_dims = 2
        
    def sample(self, num_samples):
        """Sample from the two moons distribution (similar to normflows implementation)."""
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
        """Log probability of two moons distribution (matches normflows implementation)."""
        a = torch.abs(z[:, 0])
        log_prob = (
            -0.5 * ((torch.norm(z, dim=1) - 2) / 0.2) ** 2
            - 0.5 * ((a - 2) / 0.3) ** 2
            + torch.log(1 + torch.exp(-4 * a / 0.09))
        )
        return log_prob

def target_distribution(target, device, dataset_type="2d"):
    """
    Create a grid for visualizing the distribution
    
    Args:
        target: Target distribution
        device: Device (CPU/GPU)
        dataset_type: Type of dataset
    
    Returns:
        For 2D distributions: xx, yy, zz tensors for grid visualization
        For high-dim distributions: None
    """
    if dataset_type != "2d":
        return None, None, None
    
    grid_size = 200
    xx, yy = torch.meshgrid(torch.linspace(-4, 4, grid_size), torch.linspace(-4, 4, grid_size))
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    zz = zz.to(device)

    log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    return xx, yy, zz

def train_model(model, target, device, dataset_type="2d", xx=None, yy=None, zz=None, 
                max_iter=4000, num_samples=512, show_iter=500, save_path=None,
                show_progress=True):
    """
    Train the NICE model
    
    Args:
        model: NICE model
        target: Target distribution
        device: Device (CPU/GPU)
        dataset_type: Type of dataset
        xx, yy, zz: Grid tensors for 2D visualization
        max_iter: Maximum iterations
        num_samples: Number of samples per batch
        show_iter: Interval for showing progress
        save_path: Path to save trained model
        show_progress: Whether to show progress plots
    
    Returns:
        model: Trained model
        loss_hist: Loss history
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
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        
        # Display progress
        if show_progress and (it + 1) % show_iter == 0:
            print(f"Iteration {it+1}, loss: {loss.item():.4f}")
            
            # Visualize 2D distributions
            if dataset_type == "2d" and xx is not None and yy is not None and zz is not None:
                model.eval()
                with torch.no_grad():
                    samples = zz.view(-1, 2)
                    log_prob = model.log_prob(samples)
                model.train()
                
                prob = torch.exp(log_prob.to('cpu').view(*xx.shape))
                prob[torch.isnan(prob)] = 0

                plt.figure(figsize=(10, 10))
                plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
                plt.gca().set_aspect('equal', 'box')
                plt.title(f"Learned distribution - Iteration {it+1}")
                plt.show()
            
            # Visualize MNIST samples
            elif dataset_type == "mnist":
                model.eval()
                with torch.no_grad():
                    # Generate samples
                    samples = model.sample(25)
                model.train()
                
                if show_progress:
                    # Reshape and visualize
                    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
                    for i, ax in enumerate(axes.flat):
                        if i < len(samples):
                            # Reshape and normalize
                            digit = samples[i].cpu().view(28, 28).detach().numpy()
                            digit = (digit + 1) / 2  # Scale from [-1, 1] to [0, 1]
                            
                            # Display
                            ax.imshow(digit, cmap='gray')
                            ax.axis('off')
                    
                    plt.tight_layout()
                    plt.suptitle(f"Generated MNIST Samples - Iteration {it+1}")
                    plt.show()

    # Save model if path is provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.save(save_path)
        print(f"Model saved to {save_path}")

    # Plot final loss history if requested
    if show_progress:
        plt.figure(figsize=(10, 5))
        plt.plot(loss_hist, label='loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss History')
        plt.legend()
        plt.show()

    return model, loss_hist

def plot_target_distribution(model, target, xx, yy, zz):
    """Plot 2D target and learned distributions side by side"""
    f, ax = plt.subplots(1, 2, sharey=True, figsize=(15, 7))

    # Plot target distribution
    log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    ax[0].pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
    ax[0].set_aspect('equal', 'box')
    ax[0].set_axis_off()
    ax[0].set_title('Target', fontsize=24)

    # Plot learned distribution
    model.eval()
    with torch.no_grad():
        log_prob = model.log_prob(zz.view(-1, 2)).to('cpu').view(*xx.shape)
    model.train()
    
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    ax[1].pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
    ax[1].set_aspect('equal', 'box')
    ax[1].set_axis_off()
    ax[1].set_title('NICE', fontsize=24)

    plt.subplots_adjust(wspace=0.1)
    plt.show()

def generate_mnist_samples(model, n_samples=25, grid_size=5, save_path=None):
    """
    Generate and visualize MNIST samples
    
    Args:
        model: Trained NICE model
        n_samples: Number of samples to generate
        grid_size: Size of the visualization grid (grid_size x grid_size)
        save_path: Path to save the visualization image
    
    Returns:
        samples: Generated samples
    """
    model.eval()
    with torch.no_grad():
        # Generate samples
        samples = model.sample(n_samples)
    
    # Reshape and visualize
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    for i, ax in enumerate(axes.flat):
        if i < len(samples):
            # Reshape and normalize
            digit = samples[i].cpu().view(28, 28).detach().numpy()
            digit = (digit + 1) / 2  # Scale from [-1, 1] to [0, 1]
            
            # Display
            ax.imshow(digit, cmap='gray')
            ax.axis('off')
    
    plt.tight_layout()
    plt.suptitle("NICE Generated MNIST Samples")
    
    # Save if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    
    plt.show()
    return samples

def execute_custom_nice(image_path=None, dataset_type="2d"):
    """
    Execute custom NICE training with 2D distributions or images
    
    Args:
        image_path: Path to image (for image-based distribution)
        dataset_type: "2d" or "image"
    """
    num_layers = 8
    hidden_dim = 128
    model, device, target = setup_model(image_path, num_layers, hidden_dim, 
                                       dataset_type="image" if image_path else "2d")
    xx, yy, zz = target_distribution(target, device)
    model, _ = train_model(model, target, device, dataset_type, xx, yy, zz, 
                          max_iter=6000, show_iter=200)
    plot_target_distribution(model, target, xx, yy, zz)

def execute_mnist_nice(train=False, model_path="data/nice_mnist.pth"):
    """
    Execute NICE on MNIST dataset
    
    Args:
        train: Whether to train a new model or load existing
        model_path: Path to save/load model
    
    Returns:
        model: Trained model
    """
    # Set up model and target
    num_layers = 6
    hidden_dim = 1024
    model, device, target = setup_model(num_layers=num_layers, hidden_dim=hidden_dim, 
                                       dataset_type="mnist")
    
    # Train or load model
    if train:
        xx, yy, zz = None, None, None  # No 2D visualization for MNIST
        model, _ = train_model(model, target, device, dataset_type="mnist", xx=xx, yy=yy, zz=zz,
                              max_iter=10000, show_iter=500, save_path=model_path)
    else:
        try:
            model.load(model_path)
            print(f"Model loaded from {model_path}")
        except:
            print(f"Could not load model from {model_path}. Training new model...")
            xx, yy, zz = None, None, None  # No 2D visualization for MNIST
            model, _ = train_model(model, target, device, dataset_type="mnist", xx=xx, yy=yy, zz=zz,
                                  max_iter=10000, show_iter=500, save_path=model_path)
    
    # Generate and visualize samples
    save_path = os.path.join(os.path.dirname(model_path), "nice_mnist_samples.png")
    generate_mnist_samples(model, save_path=save_path)
    
    return model