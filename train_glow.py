import torch
import torchvision as tv
import numpy as np
import normflows as nf

from matplotlib import pyplot as plt
from tqdm import tqdm

import streamlit as st

def setup_model(num_classes=10, L=2, K=16, hidden_channels=256, split_mode='channel', scale=True):
    # Modification de seed pour la reproductibilité
    torch.manual_seed(3008)
    
    # MNIST est en niveaux de gris (1 canal) et 28x28
    input_shape = (1, 28, 28)
    n_dims = np.prod(input_shape)
    channels = 1  # 1 canal au lieu de 3

    # Configuration des flows, distributions et opérations de fusion
    q0 = []
    merges = []
    flows = []
    for i in range(L):
        flows_ = []
        for j in range(K):
            flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
                                        split_mode=split_mode, scale=scale)]
        flows_ += [nf.flows.Squeeze()]
        flows += [flows_]
        if i > 0:
            merges += [nf.flows.Merge()]
            latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i), 
                            input_shape[2] // 2 ** (L - i))
        else:
            latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L, 
                            input_shape[2] // 2 ** L)
        q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes)]

    # Construction du modèle flow avec l'architecture multi-échelle
    model = nf.MultiscaleFlow(q0, flows, merges)

    # Déplacement du modèle sur GPU si disponible
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    model = model.to(device)

    return model, num_classes, n_dims, device

def prepare_training_data(batch_size=128):
    # Transformation pour MNIST
    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        nf.utils.Scale(255. / 256.),
        nf.utils.Jitter(1 / 256.)
    ])
    
    # Chargement des données MNIST
    train_data = tv.datasets.MNIST('datasets/', train=True,
                                download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                           drop_last=True)

    test_data = tv.datasets.MNIST('datasets/', train=False,
                               download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    train_iter = iter(train_loader)

    return train_loader, test_loader, train_iter

def train_model(model, train_loader, train_iter, device, max_iter=20000, model_path="glow_model.pth"):
    loss_hist = np.array([])
    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3, weight_decay=1e-5)

    for i in tqdm(range(max_iter)):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)
        optimizer.zero_grad()
        loss = model.forward_kld(x.to(device), y.to(device))
            
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()

        loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())
        del(x, y, loss)
    torch.save(model.state_dict(), model_path)

    # Affichage de l'historique des pertes
    plt.figure(figsize=(10, 10))
    plt.plot(loss_hist, label='loss')
    plt.legend()
    plt.show()

def generate_samples(model, num_classes, device, num_sample=10):
    with torch.no_grad():
        y = torch.arange(num_classes).repeat(num_sample).to(device)
        x, _ = model.sample(y=y)
        x_ = torch.clamp(x, 0, 1)
        
        # Pour MNIST, on utilise une palette de gris
        plt.figure(figsize=(10, 10))
        plt.imshow(np.transpose(tv.utils.make_grid(x_, nrow=num_classes).cpu().numpy(), (1, 2, 0)), cmap='gray')
        plt.savefig("images/glow/loss_plot.png")
        plt.close()

def get_bits_per_dim(model, test_loader, n_dims, device):
    n = 0
    bpd_cum = 0
    with torch.no_grad():
        for x, y in iter(test_loader):
            nll = model(x.to(device), y.to(device))
            nll_np = nll.cpu().numpy() 
            bpd_cum += np.nansum(nll_np / np.log(2) / n_dims + 8)
            n += len(x) - np.sum(np.isnan(nll_np))
        
    print('Bits per dim: ', bpd_cum / n)

def execute_glow():
    model, num_classes, n_dims, device = setup_model()
    print("setup_model done")
    train_loader, test_loader, train_iter = prepare_training_data()
    print("prepare_training_data done")
    # train_model(model, train_loader, train_iter, device)
    # model, num_classes, n_dims = setup_model()
    model.load_state_dict(torch.load('glow_model.pth', map_location='cpu'))
    print("load model done")
    model.eval()
    print("model.eval done")
    generate_samples(model, num_classes, device)
    print("generate_samples done")
    # get_bits_per_dim(model, test_loader, n_dims, device)
    # print("get_bits_per_dim done")


# import torch
# import torchvision as tv
# import numpy as np
# import normflows as nf

# from matplotlib import pyplot as plt
# from tqdm import tqdm

# def setup_model(num_classes=10, L=3, K=16, hidden_channels=256, split_mode='channel', scale=True):

#     torch.manual_seed(3008)
#     input_shape = (3, 32, 32)
#     n_dims = np.prod(input_shape)
#     channels = 3

#     # Set up flows, distributions and merge operations
#     q0 = []
#     merges = []
#     flows = []
#     for i in range(L):
#         flows_ = []
#         for j in range(K):
#             flows_ += [nf.flows.GlowBlock(channels * 2 ** (L + 1 - i), hidden_channels,
#                                         split_mode=split_mode, scale=scale)]
#         flows_ += [nf.flows.Squeeze()]
#         flows += [flows_]
#         if i > 0:
#             merges += [nf.flows.Merge()]
#             latent_shape = (input_shape[0] * 2 ** (L - i), input_shape[1] // 2 ** (L - i), 
#                             input_shape[2] // 2 ** (L - i))
#         else:
#             latent_shape = (input_shape[0] * 2 ** (L + 1), input_shape[1] // 2 ** L, 
#                             input_shape[2] // 2 ** L)
#         q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes)]

#     # Construct flow model with the multiscale architecture
#     model = nf.MultiscaleFlow(q0, flows, merges)

#     # Move model on GPU if available
#     enable_cuda = True
#     device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
#     model = model.to(device)

#     return model, num_classes, n_dims, device

# def prepare_training_data(batch_size = 128):

#     transform = tv.transforms.Compose([tv.transforms.ToTensor(), nf.utils.Scale(255. / 256.), nf.utils.Jitter(1 / 256.)])
#     train_data = tv.datasets.CIFAR10('datasets/', train=True,
#                                     download=True, transform=transform)
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
#                                             drop_last=True)

#     test_data = tv.datasets.CIFAR10('datasets/', train=False,
#                                     download=True, transform=transform)
#     test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

#     train_iter = iter(train_loader)

#     return train_loader, test_loader, train_iter

# def train_model(model, train_loader, train_iter, device, max_iter=20000):

#     loss_hist = np.array([])
#     optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3, weight_decay=1e-5)

#     for i in tqdm(range(max_iter)):
#         try:
#             x, y = next(train_iter)
#         except StopIteration:
#             train_iter = iter(train_loader)
#             x, y = next(train_iter)
#         optimizer.zero_grad()
#         loss = model.forward_kld(x.to(device), y.to(device))
            
#         if ~(torch.isnan(loss) | torch.isinf(loss)):
#             loss.backward()
#             optimizer.step()

#         loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())
#         del(x, y, loss)

#     # plt.figure(figsize=(10, 10))
#     # plt.plot(loss_hist, label='loss')
#     # plt.legend()
#     # plt.show()

# def generate_samples(model, num_classes, device, num_sample=10):

#     with torch.no_grad():
#         y = torch.arange(num_classes).repeat(num_sample).to(device)
#         x, _ = model.sample(y=y)
#         x_ = torch.clamp(x, 0, 1)
#         plt.figure(figsize=(10, 10))
#         plt.imshow(np.transpose(tv.utils.make_grid(x_, nrow=num_classes).cpu().numpy(), (1, 2, 0)))
#         plt.show()

# def get_bits_per_dim(model, test_loader, n_dims, device):
#     n = 0
#     bpd_cum = 0
#     with torch.no_grad():
#         for x, y in iter(test_loader):
#             nll = model(x.to(device), y.to(device))
#             nll_np = nll.cpu().numpy() 
#             bpd_cum += np.nansum(nll_np / np.log(2) / n_dims + 8)
#             n += len(x) - np.sum(np.isnan(nll_np))
        
#     print('Bits per dim: ', bpd_cum / n)

# def execute_glow():
#     model, num_classes, n_dims, device = setup_model()
#     train_loader, test_loader, train_iter = prepare_training_data()
#     train_model(model, train_loader, train_iter, device)
#     generate_samples(model, num_classes, device)
#     get_bits_per_dim(model, test_loader, n_dims, device)