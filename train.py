#!/usr/bin/env python

# Modified Horovod PyTorch MNIST example

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.utils import checkpoint
import horovod.torch as hvd
import graphics
from utils import ResultLogger

# Suppress verbose warnings
os.environ['PYTHONWARNINGS'] = 'ignore'


def _print(*args, **kwargs):
    if hvd.rank() == 0:
        print(*args, **kwargs)


def init_visualizations(hps, model, logdir):
    def sample_batch(y, eps):
        n_batch = hps.local_batch_train
        xs = []
        for i in range(int(np.ceil(len(eps) / n_batch))):
            batch_y = y[i*n_batch:i*n_batch + n_batch]
            batch_eps = eps[i*n_batch:i*n_batch + n_batch]
            
            # Convert to PyTorch tensors
            batch_y = torch.tensor(batch_y, dtype=torch.long).to(model.device)
            batch_eps = torch.tensor(batch_eps, dtype=torch.float32).to(model.device)
            
            with torch.no_grad():
                samples = model.sample(batch_y, batch_eps)
            xs.append(samples.cpu().numpy())
        return np.concatenate(xs)

    def draw_samples(epoch):
        if hvd.rank() != 0:
            return

        rows = 10 if hps.image_size <= 64 else 4
        cols = rows
        n_batch = rows*cols
        y = np.asarray([_y % hps.n_y for _y in (
            list(range(cols)) * rows)], dtype='int32')

        # temperatures = [0., .25, .5, .626, .75, .875, 1.] #previously
        temperatures = [0., .25, .5, .6, .7, .8, .9, 1.]

        x_samples = []
        x_samples.append(sample_batch(y, [.0]*n_batch))
        x_samples.append(sample_batch(y, [.25]*n_batch))
        x_samples.append(sample_batch(y, [.5]*n_batch))
        x_samples.append(sample_batch(y, [.6]*n_batch))
        x_samples.append(sample_batch(y, [.7]*n_batch))
        x_samples.append(sample_batch(y, [.8]*n_batch))
        x_samples.append(sample_batch(y, [.9] * n_batch))
        x_samples.append(sample_batch(y, [1.]*n_batch))

        for i in range(len(x_samples)):
            x_sample = np.reshape(
                x_samples[i], (n_batch, hps.image_size, hps.image_size, 3))
            graphics.save_raster(x_sample, logdir +
                                 'epoch_{}_sample_{}.png'.format(epoch, i))

    return draw_samples


# ===
# Code for getting data
# ===
def get_data(hps):
    if hps.image_size == -1:
        hps.image_size = {'mnist': 32, 'cifar10': 32, 'imagenet-oord': 64,
                          'imagenet': 256, 'celeba': 256, 'lsun_realnvp': 64, 'lsun': 256}[hps.problem]
    if hps.n_test == -1:
        hps.n_test = {'mnist': 10000, 'cifar10': 10000, 'imagenet-oord': 50000, 'imagenet': 50000,
                      'celeba': 3000, 'lsun_realnvp': 300*hvd.size(), 'lsun': 300*hvd.size()}[hps.problem]
    hps.n_y = {'mnist': 10, 'cifar10': 10, 'imagenet-oord': 1000,
               'imagenet': 1000, 'celeba': 1, 'lsun_realnvp': 1, 'lsun': 1}[hps.problem]
    if hps.data_dir == "":
        hps.data_dir = {'mnist': None, 'cifar10': None, 'imagenet-oord': '/mnt/host/imagenet-oord-tfr', 'imagenet': '/mnt/host/imagenet-tfr',
                        'celeba': '/mnt/host/celeba-reshard-tfr', 'lsun_realnvp': '/mnt/host/lsun_realnvp', 'lsun': '/mnt/host/lsun'}[hps.problem]

    if hps.problem == 'lsun_realnvp':
        hps.rnd_crop = True
    else:
        hps.rnd_crop = False

    if hps.category:
        hps.data_dir += ('/%s' % hps.category)

    # Use anchor_size to rescale batch size based on image_size
    s = hps.anchor_size
    hps.local_batch_train = hps.n_batch_train * \
        s * s // (hps.image_size * hps.image_size)
    hps.local_batch_test = {64: 50, 32: 25, 16: 10, 8: 5, 4: 2, 2: 2, 1: 1}[
        hps.local_batch_train]  # round down to closest divisor of 50
    hps.local_batch_init = hps.n_batch_init * \
        s * s // (hps.image_size * hps.image_size)

    print("Rank {} Batch sizes Train {} Test {} Init {}".format(
        hvd.rank(), hps.local_batch_train, hps.local_batch_test, hps.local_batch_init))

    if hps.problem in ['imagenet-oord', 'imagenet', 'celeba', 'lsun_realnvp', 'lsun']:
        hps.direct_iterator = True
        import data_loaders.get_data_pytorch as v
        train_loader, test_loader, init_loader = \
            v.get_data(hps.data_dir, hvd.size(), hvd.rank(), hps.pmap, hps.fmap, hps.local_batch_train,
                       hps.local_batch_test, hps.local_batch_init, hps.image_size, hps.rnd_crop)

    elif hps.problem in ['mnist', 'cifar10']:
        hps.direct_iterator = False
        import data_loaders.get_mnist_cifar_pytorch as v
        train_loader, test_loader, init_loader = \
            v.get_data(hps.problem, hvd.size(), hvd.rank(), hps.dal, hps.local_batch_train,
                       hps.local_batch_test, hps.local_batch_init, hps.image_size)

    else:
        raise Exception("Unknown problem: {}".format(hps.problem))

    return train_loader, test_loader, init_loader


def process_results(results):
    stats = ['loss', 'bits_x', 'bits_y', 'pred_loss']
    assert len(stats) == results.shape[0]
    res_dict = {}
    for i in range(len(stats)):
        res_dict[stats[i]] = "{:.4f}".format(results[i])
    return res_dict


def main(hps):
    # Initialize Horovod
    hvd.init()

    # Set random seeds
    torch.manual_seed(hvd.rank() + hvd.size() * hps.seed)
    np.random.seed(hvd.rank() + hvd.size() * hps.seed)
    
    # Set up device
    torch.cuda.set_device(hvd.local_rank())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get data and set train_its and valid_its
    train_loader, test_loader, init_loader = get_data(hps)
    hps.train_its, hps.test_its, hps.full_test_its = get_its(hps)

    # Create log dir
    logdir = os.path.abspath(hps.logdir) + "/"
    if hvd.rank() == 0:
        if not os.path.exists(logdir):
            os.makedirs(logdir)

    # Create model
    import model_pytorch as model_module
    model = model_module.Model(hps, device)
    model.to(device)

    # Set up optimizer
    if hps.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=hps.lr, betas=(hps.beta1, 0.999))
    elif hps.optimizer == 'adamax':
        optimizer = optim.Adamax(model.parameters(), lr=hps.lr, betas=(hps.beta1, 0.999))
    else:
        raise ValueError("Unknown optimizer: {}".format(hps.optimizer))

    # Add Horovod Distributed Optimizer
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters())

    # Broadcast parameters from rank 0 to all other processes
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    # Initialize visualization functions
    visualise = init_visualizations(hps, model, logdir)

    if not hps.inference:
        # Perform training
        train(model, optimizer, train_loader, test_loader, device, hps, logdir, visualise)
    else:
        infer(model, test_loader, device, hps)


def infer(model, test_loader, device, hps):
    # Example of using model in inference mode
    model.eval()
    xs = []
    zs = []
    
    for it, (x, y) in enumerate(test_loader):
        if it >= hps.full_test_its:
            break
            
        x = x.to(device)
        y = y.to(device)
        
        with torch.no_grad():
            z = model.encode(x, y)
            x_recon = model.decode(y, z)
            
        xs.append(x_recon.cpu().numpy())
        zs.append(z.cpu().numpy())

    x = np.concatenate(xs, axis=0)
    z = np.concatenate(zs, axis=0)
    
    if hvd.rank() == 0:
        os.makedirs('logs', exist_ok=True)
        np.save('logs/x.npy', x)
        np.save('logs/z.npy', z)
    
    return zs


def train(model, optimizer, train_loader, test_loader, device, hps, logdir, visualise):
    _print(hps)
    _print('Starting training. Logging to', logdir)
    _print('epoch n_processed n_images ips dtrain dtest dsample dtot train_results test_results msg')

    # Train
    n_processed = 0
    n_images = 0
    train_time = 0.0
    test_loss_best = float('inf')

    if hvd.rank() == 0:
        train_logger = ResultLogger(logdir + "train.txt", **hps.__dict__)
        test_logger = ResultLogger(logdir + "test.txt", **hps.__dict__)

    # For EMA (Exponential Moving Average)
    ema_model = None
    if hps.polyak_epochs > 0:
        ema_model = model_module.create_ema_model(model)

    tcurr = time.time()
    for epoch in range(1, hps.epochs):
        t = time.time()
        model.train()
        train_results = []

        for it, (x, y) in enumerate(train_loader):
            if it >= hps.train_its:
                break
                
            # Set learning rate, linearly annealed from 0 in the first hps.epochs_warmup epochs
            lr = hps.lr * min(1., n_processed / (hps.n_train * hps.epochs_warmup))
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Get data
            x = x.to(device)
            y = y.to(device)

            # Forward and backward passes
            optimizer.zero_grad()
            loss, results = model(x, y)
            loss.backward()
            optimizer.step()

            # Update EMA model if enabled
            if ema_model is not None:
                model_module.update_ema_model(ema_model, model)

            # Log results
            train_results.append(results.cpu().numpy())
            
            if hps.verbose and hvd.rank() == 0:
                _print(n_processed, time.time() - t, train_results[-1])
                sys.stdout.flush()

            # Images seen wrt anchor resolution
            n_processed += hvd.size() * hps.n_batch_train
            # Actual images seen at current resolution
            n_images += hvd.size() * hps.local_batch_train

        train_results = np.mean(np.asarray(train_results), axis=0)

        dtrain = time.time() - t
        ips = (hps.train_its * hvd.size() * hps.local_batch_train) / dtrain
        train_time += dtrain

        if hvd.rank() == 0:
            train_logger.log(epoch=epoch, n_processed=n_processed, n_images=n_images, train_time=int(
                train_time), **process_results(train_results))

        if epoch < 10 or (epoch < 50 and epoch % 10 == 0) or epoch % hps.epochs_full_valid == 0:
            test_results = []
            msg = ''

            t = time.time()
            model.eval()

            if epoch % hps.epochs_full_valid == 0:
                # Full validation run
                with torch.no_grad():
                    for it, (x, y) in enumerate(test_loader):
                        if it >= hps.full_test_its:
                            break
                            
                        x = x.to(device)
                        y = y.to(device)
                        
                        _, results = model(x, y)
                        test_results.append(results.cpu().numpy())
                
                test_results = np.mean(np.asarray(test_results), axis=0)

                if hvd.rank() == 0:
                    test_logger.log(epoch=epoch, n_processed=n_processed,
                                    n_images=n_images, **process_results(test_results))

                    # Save checkpoint
                    if test_results[0] < test_loss_best:
                        test_loss_best = test_results[0]
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch,
                            'hps': hps,
                        }, logdir + "model_best_loss.pt")
                        msg += ' *'

            dtest = time.time() - t

            # Sample
            t = time.time()
            if epoch == 1 or epoch == 10 or epoch % hps.epochs_full_sample == 0:
                visualise(epoch)
            dsample = time.time() - t

            if hvd.rank() == 0:
                dcurr = time.time() - tcurr
                tcurr = time.time()
                _print(epoch, n_processed, n_images, "{:.1f} {:.1f} {:.1f} {:.1f} {:.1f}".format(
                    ips, dtrain, dtest, dsample, dcurr), train_results, test_results, msg)

    if hvd.rank() == 0:
        _print("Finished!")


# Get number of training and validation iterations
def get_its(hps):
    # These run for a fixed amount of time. As anchored batch is smaller, we've actually seen fewer examples
    train_its = int(np.ceil(hps.n_train / (hps.n_batch_train * hvd.size())))
    test_its = int(np.ceil(hps.n_test / (hps.n_batch_train * hvd.size())))
    train_epoch = train_its * hps.n_batch_train * hvd.size()

    # Do a full validation run
    if hvd.rank() == 0:
        print(hps.n_test, hps.local_batch_test, hvd.size())
    assert hps.n_test % (hps.local_batch_test * hvd.size()) == 0
    full_test_its = hps.n_test // (hps.local_batch_test * hvd.size())

    if hvd.rank() == 0:
        print("Train epoch size: " + str(train_epoch))
    return train_its, test_its, full_test_its


# Create companion model_pytorch.py file
def create_model_pytorch_file():
    """
    Helper function to create a PyTorch model file that would complement this script.
    This would typically be saved separately as model_pytorch.py
    """
    model_code = ""


class Model(nn.Module):
    def __init__(self, hps, device):
        super(Model, self).__init__()
        self.hps = hps
        self.device = device
        self.n_levels = hps.n_levels
        self.n_bits_x = hps.n_bits_x
        self.n_y = hps.n_y
        self.width = hps.width
        self.depth = hps.depth
        self.weight_y = hps.weight_y
        self.flow_permutation = hps.flow_permutation
        self.flow_coupling = hps.flow_coupling
        self.learntop = hps.learntop
        self.ycond = hps.ycond
        
        # Build the model
        self._build()
        
    def _build(self):
        # Define layers for the flow model
        self.flows = nn.ModuleList()
        for i in range(self.n_levels):
            if i == 0:
                # First level
                flow = Flow(
                    self.hps.image_size,
                    self.hps.width,
                    self.hps.depth,
                    self.flow_permutation,
                    self.flow_coupling,
                    self.hps.n_bits_x
                )
            else:
                # Additional levels
                flow = Flow(
                    self.hps.image_size // (2**i),
                    self.hps.width,
                    self.hps.depth,
                    self.flow_permutation,
                    self.flow_coupling,
                    self.hps.n_bits_x
                )
            self.flows.append(flow)
        
        # Top prior
        if self.learntop:
            self.prior = nn.Parameter(torch.zeros(1, self.hps.width, 
                                                  self.hps.image_size // (2**(self.n_levels-1)),
                                                  self.hps.image_size // (2**(self.n_levels-1))))
        else:
            self.register_buffer('prior', torch.zeros(1))
        
        # Condition on y
        if self.ycond:
            self.y_emb = nn.Embedding(self.n_y, self.width)
        
        # For gradient checkpointing
        self.use_checkpoint = self.hps.gradient_checkpointing > 0
        
    def forward(self, x, y=None):
        # Preprocess x
        x = preprocess(x, self.n_bits_x)
        
        # Initialize log likelihood
        objective = torch.zeros_like(x[:, 0, 0, 0])
        
        # Process through flow levels
        for i in range(self.n_levels):
            if self.use_checkpoint and self.training:
                objective, x = checkpoint.checkpoint(self.flows[i], x, objective)
            else:
                objective, x = self.flows[i](x, objective)
        
        # Top prior
        if self.learntop:
            prior = self.prior
        else:
            prior = torch.zeros_like(x)
            
        # Condition on y if needed
        if self.ycond:
            y_emb = self.y_emb(y)
            # Reshape and add to prior
            y_emb = y_emb.view(y_emb.shape[0], self.width, 1, 1)
            prior = prior + y_emb
        
        # Final loss calculation
        objective = objective + 0.5 * (prior - x).pow(2).sum(dim=[1, 2, 3])
        
        # Log likelihood in bits per dimension
        bpd = objective / (np.log(2.) * x.shape[1] * x.shape[2] * x.shape[3])
        
        # If we're also predicting y, add that loss
        pred_loss = torch.zeros_like(bpd)
        if self.weight_y > 0 and self.ycond:
            # Make a prediction of y from the encoding
            y_logits = self.y_prediction(x)
            pred_loss = F.cross_entropy(y_logits, y, reduction='none')
            
            # Add weighted prediction loss
            bpd = bpd + self.weight_y * pred_loss
        
        # Return results
        results = torch.stack([bpd, bpd, pred_loss, pred_loss])
        return bpd.mean(), results
        
    def encode(self, x, y=None):
        # Preprocess x
        x = preprocess(x, self.n_bits_x)
        
        # Initialize log likelihood (not used for encoding)
        objective = torch.zeros_like(x[:, 0, 0, 0])
        
        # Process through flow levels
        for i in range(self.n_levels):
            objective, x = self.flows[i](x, objective)
            
        return x
        
    def decode(self, y, z):
        # Start from latent z
        x = z
        
        # Process through flow levels in reverse
        for i in reversed(range(self.n_levels)):
            x = self.flows[i].reverse(x, y if self.ycond else None)
            
        # Postprocess
        x = postprocess(x, self.n_bits_x)
        return x
        
    def sample(self, y, eps=None):
        # Sample from the prior
        if eps is None:
            eps = torch.randn(y.shape[0], self.width, 
                              self.hps.image_size // (2**(self.n_levels-1)),
                              self.hps.image_size // (2**(self.n_levels-1)),
                              device=self.device)
        
        # Condition on y if needed
        if self.ycond:
            y_emb = self.y_emb(y)
            # Reshape and add to eps
            y_emb = y_emb.view(y_emb.shape[0], self.width, 1, 1)
            eps = eps + y_emb
            
        # Decode
        return self.decode(y, eps)
        
    def y_prediction(self, z):
        # Make a prediction of y from the encoding z
        # This is a simplified version, real implementation would depend on the model architecture
        pooled = F.adaptive_avg_pool2d(z, (1, 1)).view(z.shape[0], -1)
        return F.linear(pooled, self.y_emb.weight)
        
    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'hps': self.hps,
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('hps', None)


class Flow(nn.Module):
    def __init__(self, size, width, depth, flow_permutation, flow_coupling, n_bits_x):
        super(Flow, self).__init__()
        self.size = size
        self.width = width
        self.depth = depth
        self.flow_permutation = flow_permutation
        self.flow_coupling = flow_coupling
        self.n_bits_x = n_bits_x
        
        # Build the flow
        self._build()
        
    def _build(self):
        # Define coupling layers
        self.coupling_layers = nn.ModuleList()
        for i in range(self.depth):
            # Define permutation type
            if self.flow_permutation == 0:
                # Reverse
                self.coupling_layers.append(ReverseFlow())
            elif self.flow_permutation == 1:
                # Shuffle
                self.coupling_layers.append(ShuffleFlow(self.width))
            elif self.flow_permutation == 2:
                # Invertible 1x1 convolution
                self.coupling_layers.append(InvertibleConv1x1(self.width))
            
            # Define coupling type
            if self.flow_coupling == 0:
                # Additive coupling
                self.coupling_layers.append(AdditiveCoupling(self.width))
            elif self.flow_coupling == 1:
                # Affine coupling
                self.coupling_layers.append(AffineCoupling(self.width))
        
    def forward(self, x, objective):
        # Forward pass through all coupling layers
        for coupling_layer in self.coupling_layers:
            x, objective = coupling_layer(x, objective)
        return objective, x
        
    def reverse(self, x, y=None):
        # Reverse pass through coupling layers
        for coupling_layer in reversed(self.coupling_layers):
            x = coupling_layer.reverse(x)
        return x


class ReverseFlow(nn.Module):
    def __init__(self):
        super(ReverseFlow, self).__init__()
        
    def forward(self, x, objective):
        # Reverse the order of channels
        return x.flip(1), objective
        
    def reverse(self, x):
        # Reverse operation is the same
        return x.flip(1)


class ShuffleFlow(nn.Module):
    def __init__(self, num_channels):
        super(ShuffleFlow, self).__init__()
        self.register_buffer('perm', torch.randperm(num_channels))
        self.register_buffer('inv_perm', torch.argsort(self.perm))
        
    def forward(self, x, objective):
        # Shuffle channels according to permutation
        return x[:, self.perm], objective
        
    def reverse(self, x):
        # Unshuffle channels
        return x[:, self.inv_perm]


class InvertibleConv1x1(nn.Module):
    def __init__(self, num_channels):
        super(InvertibleConv1x1, self).__init__()
        # Initialize as a random rotation matrix
        w_init = torch.qr(torch.randn(num_channels, num_channels))[0]
        self.weight = nn.Parameter(w_init)
        
    def forward(self, x, objective):
        # Get batch size
        batch_size, _, h, w = x.shape
        
        # Calculate log determinant
        log_det = torch.slogdet(self.weight)[1] * h * w
        
        # Apply conv
        weight = self.weight.view(self.weight.shape[0], self.weight.shape[1], 1, 1)
        z = F.conv2d(x, weight)
        
        # Update objective
        objective = objective + log_det
        
        return z, objective
        
    def reverse(self, x):
        # Get inverse weight
        weight_inv = torch.inverse(self.weight)
        weight_inv = weight_inv.view(weight_inv.shape[0], weight_inv.shape[1], 1, 1)
        
        # Apply inverse conv
        z = F.conv2d(x, weight_inv)
        
        return z


class AdditiveCoupling(nn.Module):
    def __init__(self, num_channels):
        super(AdditiveCoupling, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels // 2, num_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels // 2, kernel_size=3, padding=1)
        )
        
    def forward(self, x, objective):
        # Split input
        x1, x2 = torch.chunk(x, 2, dim=1)
        
        # Calculate shift
        shift = self.net(x1)
        
        # Apply shift
        y1 = x1
        y2 = x2 + shift
        
        # Combine outputs
        y = torch.cat([y1, y2], dim=1)
        
        return y, objective
        
    def reverse(self, y):
        # Split input
        y1, y2 = torch.chunk(y, 2, dim=1)
        
        # Calculate shift
        shift = self.net(y1)
        
        # Apply inverse shift
        x1 = y1
        x2 = y2 - shift
        
        # Combine outputs
        x = torch.cat([x1, x2], dim=1)
        
        return x


class AffineCoupling(nn.Module):
    def __init__(self, num_channels):
        super(AffineCoupling, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_channels // 2, num_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        )
        
    def forward(self, x, objective):
        # Split input
        x1, x2 = torch.chunk(x, 2, dim=1)
        
        # Calculate scale and shift
        h = self.net(x1)
        shift = h[:, :h.shape[1]//2]
        # Add sigmoid to ensure scale is positive
        scale = torch.sigmoid(h[:, h.shape[1]//2:]) + 0.5
        
        # Apply affine transformation
        y1 = x1
        y2 = x2 * scale + shift
        
        # Update objective with log determinant
        objective = objective + torch.sum(torch.log(scale), dim=[1, 2, 3])
        
        # Combine outputs
        y = torch.cat([y1, y2], dim=1)
        
        return y, objective
        
    def reverse(self, y):
        # Split input
        y1, y2 = torch.chunk(y, 2, dim=1)
        
        # Calculate scale and shift
        h = self.net(y1)
        shift = h[:, :h.shape[1]//2]
        # Add sigmoid to ensure scale is positive
        scale = torch.sigmoid(h[:, h.shape[1]//2:]) + 0.5
        
        # Apply inverse affine transformation
        x1 = y1
        x2 = (y2 - shift) / scale
        
        # Combine outputs
        x = torch.cat([x1, x2], dim=1)
        
        return x


# Preprocessing and postprocessing functions
def preprocess(x, n_bits_x=8):
    """
    Preprocess the input data to the range expected by the model
    """
    # Scale from [0, 2^n_bits - 1] to [0, 1]
    x = x / (2**n_bits_x - 1)
    
    # Scale to [-0.5, 0.5]
    x = x - 0.5
    
    # Add uniform noise
    if x.requires_grad:
        # In training mode, add noise
        noise = torch.rand_like(x) / (2**n_bits_x)
        x = x + noise - 0.5 / (2**n_bits_x)
    
    return x


def postprocess(x, n_bits_x=8):
    """
    Convert the output of the model back to image range
    """
    # Scale from [-0.5, 0.5] to [0, 1]
    x = x + 0.5
    
    # Clip to [0, 1]
    x = torch.clamp(x, 0, 1)
    
    # Scale to [0, 2^n_bits - 1]
    x = x * (2**n_bits_x - 1)
    
    return x


# EMA utilities
def create_ema_model(model):
    """
    Create an Exponential Moving Average version of the model
    """
    ema_model = type(model)(model.hps, model.device)
    ema_model.to(model.device)
    ema_model.eval()
    
    # Copy current model parameters
    for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
        param_k.data.copy_(param_q.data)
        param_k.requires_grad = False
    
    return ema_model


def update_ema_model(ema_model, model, decay=0.999):
    """
    Update the EMA model parameters
    """
    for param_q, param_k in zip(model.parameters(), ema_model.parameters()):
        param_k.data.copy_(decay * param_k.data + (1 - decay) * param_q.data)

if __name__ == "__main__":
    # This enables a ctr-C without triggering errors
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--restore_path", type=str, default='',
                        help="Location of checkpoint to restore")
    parser.add_argument("--inference", action="store_true",
                        help="Use in inference mode")
    parser.add_argument("--logdir", type=str,
                        default='./logs', help="Location to save logs")

    # Dataset hyperparams:
    parser.add_argument("--problem", type=str, default='cifar10',
                        help="Problem (mnist/cifar10/imagenet")
    parser.add_argument("--category", type=str,
                        default='', help="LSUN category")
    parser.add_argument("--data_dir", type=str, default='',
                        help="Location of data")
    parser.add_argument("--dal", type=int, default=1,
                        help="Data augmentation level: 0=None, 1=Standard, 2=Extra")

    # New dataloader params
    parser.add_argument("--fmap", type=int, default=1,
                        help="# Threads for parallel file reading")
    parser.add_argument("--pmap", type=int, default=16,
                        help="# Threads for parallel map")

    # Optimization hyperparams:
    parser.add_argument("--n_train", type=int,
                        default=50000, help="Train epoch size")
    parser.add_argument("--n_test", type=int, default=-
                        1, help="Valid epoch size")
    parser.add_argument("--n_batch_train", type=int,
                        default=64, help="Minibatch size")
    parser.add_argument("--n_batch_test", type=int,
                        default=50, help="Minibatch size")
    parser.add_argument("--n_batch_init", type=int, default=256,
                        help="Minibatch size for data-dependent init")
    parser.add_argument("--optimizer", type=str,
                        default="adamax", help="adam or adamax")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Base learning rate")
    parser.add_argument("--beta1", type=float, default=.9, help="Adam beta1")
    parser.add_argument("--polyak_epochs", type=float, default=1,
                        help="Nr of averaging epochs for Polyak and beta2")
    parser.add_argument("--weight_decay", type=float, default=1.,
                        help="Weight decay. Switched off by default.")
    parser.add_argument("--epochs", type=int, default=1000000,
                        help="Total number of training epochs")
    parser.add_argument("--epochs_warmup", type=int,
                        default=10, help="Warmup epochs")
    parser.add_argument("--epochs_full_valid", type=int,
                        default=50, help="Epochs between valid")
    parser.add_argument("--gradient_checkpointing", type=int,
                        default=1, help="Use memory saving gradients")

    # Model hyperparams:
    parser.add_argument("--image_size", type=int,
                        default=-1, help="Image size")
    parser.add_argument("--anchor_size", type=int, default=32,
                        help="Anchor size for deciding batch size")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=32,
                        help="Depth of network")
    parser.add_argument("--weight_y", type=float, default=0.00,
                        help="Weight of log p(y|x) in weighted loss")
    parser.add_argument("--n_bits_x", type=int, default=8,
                        help="Number of bits of x")
    parser.add_argument("--n_levels", type=int, default=3,
                        help="Number of levels")

    # Synthesis/Sampling hyperparameters:
    parser.add_argument("--n_sample", type=int, default=1,
                        help="minibatch size for sample")
    parser.add_argument("--epochs_full_sample", type=int,
                        default=50, help="Epochs between full scale sample")

    # Ablation
    parser.add_argument("--learntop", action="store_true",
                        help="Learn spatial prior")
    parser.add_argument("--ycond", action="store_true",
                        help="Use y conditioning")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--flow_permutation", type=int, default=2,
                        help="Type of flow. 0=reverse (realnvp), 1=shuffle, 2=invconv (ours)")
    parser.add_argument("--flow_coupling", type=int, default=0,
                        help="Coupling type: 0=additive, 1=affine")

    hps = parser.parse_args()  # So error if typo
    main(hps)