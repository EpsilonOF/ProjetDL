import torch
import numpy as np
import normflows as nf
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from tqdm import tqdm

class ImageDistribution(nf.distributions.Target):
    def __init__(self, image_path, resolution=200, blur_sigma=1.0, scale=6.0):
        super().__init__()
        try:
            # Print pour vérifier que l'image est chargée
            print(f"Chargement de l'image: {image_path}")
            img = Image.open(image_path).convert('L')
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
            x = np.linspace(-scale/2, scale/2, resolution)
            y = np.linspace(-scale/2, scale/2, resolution)
            self.xx, self.yy = np.meshgrid(x, y)
            
            # Convertir en tenseur
            self.prob_tensor = torch.tensor(self.prob, dtype=torch.float32)
            
            # Vérifier la validité de la distribution
            print(f"Somme des probabilités: {self.prob.sum()}")
            print(f"Min/Max des probabilités: {self.prob.min()}/{self.prob.max()}")
            
            # Visualiser la distribution
            plt.figure(figsize=(8, 8))
            plt.imshow(self.prob, cmap='viridis')
            plt.colorbar()
            plt.title("Distribution cible")
            plt.show()
            
        except Exception as e:
            print(f"Erreur lors du chargement de l'image: {e}")
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
    base = nf.distributions.base.DiagGaussian(2)

    # Définir la liste des flux
    flows = []
    for i in range(num_layers):
        param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        flows.append(nf.flows.Permute(2, mode='swap'))
        
    # Construire le modèle de flux
    model = nf.NormalizingFlow(base, flows)

    # Déplacer le modèle sur GPU si disponible
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
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
    grid_size = 200
    xx, yy = torch.meshgrid(torch.linspace(-3, 3, grid_size), torch.linspace(-3, 3, grid_size))
    zz = torch.cat([xx.unsqueeze(2), yy.unsqueeze(2)], 2).view(-1, 2)
    zz = zz.to(device)

    log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    return xx, yy, zz

def train_model(model, target, xx, yy, zz, device, max_iter=4000, num_samples=2**9, show_iter=500):
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
        loss_hist = np.append(loss_hist, loss.to('cpu').data.numpy())
        
        # Afficher la distribution apprise
        if (it + 1) % show_iter == 0:
            print(f"Itération {it+1}, perte: {loss.item():.4f}")
            model.eval()
            log_prob = model.log_prob(zz)
            model.train()
            prob = torch.exp(log_prob.to('cpu').view(*xx.shape))
            prob[torch.isnan(prob)] = 0

            plt.figure(figsize=(15, 15))
            plt.pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
            plt.gca().set_aspect('equal', 'box')
            plt.title(f"Distribution apprise - Itération {it+1}")
            plt.show()

    # Tracer la perte
    plt.figure(figsize=(10, 5))
    plt.plot(loss_hist, label='perte')
    plt.xlabel('Itération')
    plt.ylabel('Perte')
    plt.title('Historique de la perte')
    plt.legend()
    plt.show()

    return model

def plot_target_distribution(model, target, xx, yy, zz):
    f, ax = plt.subplots(1, 2, sharey=True, figsize=(15, 7))

    log_prob = target.log_prob(zz).to('cpu').view(*xx.shape)
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    ax[0].pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
    ax[0].set_aspect('equal', 'box')
    ax[0].set_axis_off()
    ax[0].set_title('Cible', fontsize=24)

    # Tracer la distribution apprise
    model.eval()
    log_prob = model.log_prob(zz).to('cpu').view(*xx.shape)
    model.train()
    prob = torch.exp(log_prob)
    prob[torch.isnan(prob)] = 0

    ax[1].pcolormesh(xx, yy, prob.data.numpy(), cmap='coolwarm')
    ax[1].set_aspect('equal', 'box')
    ax[1].set_axis_off()
    ax[1].set_title('Real NVP', fontsize=24)

    plt.subplots_adjust(wspace=0.1)
    plt.show()

def execute_custom_realnvp(image_path):
    num_layers = 32
    model, device, target = setup_model(image_path, num_layers)
    xx, yy, zz = target_distribution(target, device)
    model = train_model(model, target, xx, yy, zz, device, max_iter=6000, show_iter=200)
    plot_target_distribution(model, target, xx, yy, zz)
