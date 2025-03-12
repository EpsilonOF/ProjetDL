import torch
import numpy as np
import normflows as nf
from tqdm import tqdm
from matplotlib import pyplot as plt

# Set up model

# Define 2D Gaussian base distribution
def setup_model(num_layers=32):
    base = nf.distributions.base.DiagGaussian(2)

    # Define list of flows
    flows = []
    for i in range(num_layers):
        # Neural network with two hidden layers having 64 units each
        # Last layer is initialized by zeros making training more stable
        param_map = nf.nets.MLP([1, 64, 64, 2], init_zeros=True)
        # Add flow layer
        flows.append(nf.flows.AffineCouplingBlock(param_map))
        # Swap dimensions
        flows.append(nf.flows.Permute(2, mode='swap'))
        
    # Construct flow model
    model = nf.NormalizingFlow(base, flows)

    # Move model on GPU if available
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    model = model.to(device)

    return model, device


