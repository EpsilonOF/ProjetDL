import torch
import numpy as np
import normflows as nf
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from tqdm import tqdm
import streamlit as st


class ImageDistribution(nf.distributions.Target):
    """
    Distribution based on an image for 2D density estimation.

    This class converts a grayscale image into a probability distribution that can be
    used as a target for training normalizing flows. The image intensity is treated as
    a probability density, with appropriate normalization and smoothing.

    Attributes:
        prob (numpy.ndarray): Normalized probability distribution derived from the image
        resolution (int): Resolution of the discretized distribution
        scale (float): Scale factor for the coordinate system
        xx (numpy.ndarray): X-coordinates of the grid points
        yy (numpy.ndarray): Y-coordinates of the grid points
        prob_tensor (torch.Tensor): Tensor version of the probability distribution
    """

    def __init__(self, image_path, resolution=200, blur_sigma=1.0, scale=6.0):
        """
        Initialize the image-based distribution.

        Args:
            image_path (str): Path to the image file
            resolution (int, optional): Resolution of the discretized distribution.
                Defaults to 200.
            blur_sigma (float, optional): Standard deviation for Gaussian smoothing.
                Defaults to 1.0.
            scale (float, optional): Scale factor for the coordinate system.
                Defaults to 6.0.
        """
        super().__init__()
        try:
            # Print pour vérifier que l'image est chargée
            print(f"Chargement de l'image: {image_path}")
            img = Image.open(image_path).convert("L")
            print(f"Image chargée, dimensions: {img.size}")

            # Redimensionner et traiter l'image
            img = img.resize((resolution, resolution))
            self.prob = np.array(img).astype(np.float32)

            # Inverser si nécessaire (pour les images où le fond est blanc)
            if self.prob.mean() > 127:
                self.prob = 255 - self.prob

            # Normaliser
            self.prob = self.prob / (self.prob.sum() + 1e-10)

            # Lisser
            self.prob = gaussian_filter(self.prob, sigma=blur_sigma)
            self.prob = self.prob / (self.prob.sum() + 1e-10)

            # Créer une grille de coordonnées
            self.resolution = resolution
            self.scale = scale
            x = np.linspace(-scale / 2, scale / 2, resolution)
            y = np.linspace(-scale / 2, scale / 2, resolution)
            self.xx, self.yy = np.meshgrid(x, y)

            # Convertir en tenseur
            self.prob_tensor = torch.tensor(self.prob, dtype=torch.float32)

            # Vérifier la validité de la distribution
            print(f"Somme des probabilités: {self.prob.sum()}")
            print(f"Min/Max des probabilités: {self.prob.min()}/{self.prob.max()}")

            # Visualiser la distribution
            plt.figure(figsize=(8, 8))
            plt.imshow(self.prob, cmap="viridis")
            plt.colorbar()
            plt.title("Distribution cible")
            plt.show()

        except Exception as e:
            print(f"Erreur lors du chargement de l'image: {e}")
            raise

    def log_prob(self, z):
        """
        Compute the log probability of samples under the image-based distribution.

        This method evaluates the log probability of points in 2D space by mapping them
        to pixels in the discretized image-based distribution.

        Args:
            z (torch.Tensor): Input tensor of shape [batch_size, 2] representing points in 2D space

        Returns:
            torch.Tensor: Log probability for each input sample of shape [batch_size]
        """
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
        """
        Generate samples from the image-based distribution.

        This method uses rejection sampling to generate points following the
        probability distribution defined by the image.

        Args:
            num_samples (int): Number of samples to generate

        Returns:
            torch.Tensor: Samples from the distribution of shape [num_samples, 2]
        """
        # Créer une distribution de probabilité aplatie pour l'échantillonnage
        flat_prob = self.prob.flatten()
        indices = np.random.choice(len(flat_prob), size=num_samples, p=flat_prob)

        # Convertir les indices en coordonnées 2D
        y_idx = indices // self.resolution
        x_idx = indices % self.resolution

        # Convertir en espace continu
        x = (x_idx / self.resolution - 0.5) * self.scale
        y = (y_idx / self.resolution - 0.5) * self.scale

        samples = np.column_stack([x, y])
        return torch.tensor(samples, dtype=torch.float32)


def setup_model(image_path=None, num_layers=32):
    """
    Set up the RealNVP model and target distribution.

    This function configures a RealNVP model with the specified number of coupling layers
    and initializes an appropriate target distribution.

    Args:
        image_path (str, optional): Path to image to use as distribution. Defaults to None.
        num_layers (int, optional): Number of coupling layers. Defaults to 32.

    Returns:
        tuple: A tuple containing:
            - model (nf.NormalizingFlow): The configured RealNVP model
            - device (torch.device): Device for tensor operations (CPU/GPU)
            - target: Target distribution object (TwoMoons or ImageDistribution)
    """
    base = nf.distributions.base.DiagGaussian(2)

    # Définir la liste des flux
    flows = []
    for i in range(num_layers):
        param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        flows.append(nf.flows.Permute(2, mode="swap"))

    # Construire le modèle de flux
    model = nf.NormalizingFlow(base, flows)

    # Déplacer le modèle sur GPU si disponible
    enable_cuda = True
    device = torch.device(
        "cuda" if torch.cuda.is_available() and enable_cuda else "cpu"
    )
    model = model.to(device)

    # Créer la distribution cible
    if image_path:
        print(f"Utilisation de l'image {image_path} comme distribution cible")
        target = ImageDistribution(image_path)
    else:
        print("Utilisation de la distribution TwoMoons par défaut")
        target = nf.distributions.TwoMoons()

    return model, device, target


def target_distribution(target, device):
    """
    Create a grid for visualizing the target distribution.

    This function creates a 2D grid for visualizing a probability distribution,
    returning tensors that can be used for creating contour plots or heatmaps.

    Args:
        target: Target distribution object with a log_prob method
        device (torch.device): Device for tensor operations (CPU/GPU)

    Returns:
        tuple: A tuple containing:
            - xx (torch.Tensor): X-coordinates of the grid
            - yy (torch.Tensor): Y-coordinates of the grid
            - zz (torch.Tensor): Flattened grid points for evaluation
    """
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
    xx,
    yy,
    zz,
    device,
    max_iter=4000,
    num_samples=2**9,
    show_iter=500,
):
    """
    Train the RealNVP model on the target distribution.

    This function trains a RealNVP model using stochastic gradient descent on samples
    from the target distribution. It also visualizes the learning progress at regular intervals.

    Args:
        model (nf.NormalizingFlow): RealNVP model to train
        target: Target distribution object with sample method
        xx (torch.Tensor): X-coordinates of the grid for visualization
        yy (torch.Tensor): Y-coordinates of the grid for visualization
        zz (torch.Tensor): Flattened grid points for evaluation
        device (torch.device): Device for tensor operations (CPU/GPU)
        max_iter (int, optional): Maximum number of training iterations. Defaults to 4000.
        num_samples (int, optional): Number of samples per training batch. Defaults to 512 (2^9).
        show_iter (int, optional): Interval for showing visualization updates. Defaults to 500.

    Returns:
        nf.NormalizingFlow: The trained RealNVP model
    """
    loss_hist = np.array([])
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    for it in tqdm(range(max_iter)):
        optimizer.zero_grad()

        # Obtenir des échantillons d'entraînement
        x = target.sample(num_samples).to(device)

        # Calculer la perte
        loss = model.forward_kld(x)

        # Rétropropagation et étape d'optimisation
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        # Enregistrer la perte
        loss_hist = np.append(loss_hist, loss.to("cpu").data.numpy())

        # Afficher la distribution apprise
        if (it + 1) % show_iter == 0:
            print(f"Itération {it + 1}, perte: {loss.item():.4f}")
            model.eval()
            log_prob = model.log_prob(zz)
            model.train()
            prob = torch.exp(log_prob.to("cpu").view(*xx.shape))
            prob[torch.isnan(prob)] = 0

            plt.figure(figsize=(15, 15))
            plt.pcolormesh(xx, yy, prob.data.numpy(), cmap="coolwarm")
            plt.gca().set_aspect("equal", "box")
            plt.title(f"Distribution apprise - Itération {it + 1}")
            plt.show()

    # Tracer la perte
    plt.figure(figsize=(10, 5))
    plt.plot(loss_hist, label="perte")
    plt.xlabel("Itération")
    plt.ylabel("Perte")
    plt.title("Historique de la perte")
    plt.legend()
    plt.show()

    return model


def plot_target_distribution(model, target, xx, yy, zz):
    """
    Plot 2D target and learned distributions side by side.

    This function creates a visualization comparing the target distribution
    with the distribution learned by the model, displayed side by side.

    Args:
        model (nf.NormalizingFlow): Trained RealNVP model
        target: Target distribution object with log_prob method
        xx (torch.Tensor): X-coordinates of the grid
        yy (torch.Tensor): Y-coordinates of the grid
        zz (torch.Tensor): Flattened grid points for evaluation
    """
    f, ax = plt.subplots(1, 2, sharey=True, figsize=(15, 7))

    log_prob = target.log_prob(zz).to("cpu").view(*xx.shape)
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    ax[0].pcolormesh(xx, yy, prob.data.numpy(), cmap="coolwarm")
    ax[0].set_aspect("equal", "box")
    ax[0].set_axis_off()
    ax[0].set_title("Cible", fontsize=24)

    # Tracer la distribution apprise
    model.eval()
    log_prob = model.log_prob(zz).to("cpu").view(*xx.shape)
    model.train()
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    ax[1].pcolormesh(xx, yy, prob.data.numpy(), cmap="coolwarm")
    ax[1].set_aspect("equal", "box")
    ax[1].set_axis_off()
    ax[1].set_title("Real NVP", fontsize=24)

    plt.subplots_adjust(wspace=0.1)
    plt.show()


def execute_realnvp(n_blocks, max_iter):
    """
    Executes the RealNVP model to learn a target distribution.

    This function configures and trains a RealNVP model with a specified number of coupling blocks
    and a maximum number of iterations. It displays intermediate results and the final loss curve.

    Args:
        n_blocks (int): The number of coupling blocks in the model. Each block typically consists
                        of two layers, so the total number of layers will be `n_blocks * 2`.
        max_iter (int): The maximum number of iterations for training the model.

    Returns:
        None

    Description:
        - Configures the model parameters based on the specified number of blocks.
        - Initializes the RealNVP model with defined hyperparameters.
        - Trains the model over a specified number of iterations.
        - Displays thumbnails of the learned distribution at regular intervals.
        - Displays the loss curve at the end of training.

    Notes:
        - Uses the Adam optimizer with a learning rate of 5e-4 and weight decay of 1e-5.
        - Displays thumbnails of the learned distribution every `max_iter/10` iterations.
        - Handles NaN or infinite values in the loss to prevent training interruptions.
    """
    # Configuration des paramètres utilisateur
    num_layers = n_blocks * 2  # Chaque bloc de couplage est souvent pair
    show_iter = int(max_iter / 10)  # Affichage intermédiaire

    # Initialisation du modèle avec les hyperparamètres de l'utilisateur pour
    # TwoMoons
    model, device, target = setup_model(num_layers=num_layers)
    xx, yy, zz = target_distribution(target, device)

    loss_hist = []
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    cols = st.columns(
        int((max_iter // show_iter) / 2)
    )  # Colonnes pour affichage des miniatures
    cols2 = cols
    for it in range(max_iter):
        optimizer.zero_grad()
        x = target.sample(512).to(device)
        loss = model.forward_kld(x)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        loss_hist.append(loss.item())

        if (it + 1) % show_iter == 0:
            model.eval()
            with torch.no_grad():
                log_prob = model.log_prob(zz)
            prob = torch.exp(log_prob.to("cpu").view(*xx.shape))
            prob[torch.isnan(prob)] = 0

            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pcolormesh(xx, yy, prob.numpy(), cmap="coolwarm")
            ax.set_aspect("equal", "box")
            ax.axis("off")
            ax.set_title(f"Itération {it + 1}", fontsize=10)
            if it // show_iter >= len(cols):
                cols2[(it // show_iter) % len(cols)].pyplot(fig)
            else:
                cols[it // show_iter].pyplot(fig)
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
