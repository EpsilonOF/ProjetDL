# data_loaders/get_data_pytorch.py

import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from PIL import Image
import glob
import random


class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = []
        self.labels = []
        
        # Get all classes
        class_dirs = sorted(glob.glob(os.path.join(root_dir, '*')))
        for class_idx, class_dir in enumerate(class_dirs):
            img_paths = glob.glob(os.path.join(class_dir, '*.JPEG'))
            for img_path in img_paths:
                self.imgs.append(img_path)
                self.labels.append(class_idx)
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


def get_data(data_dir, n_procs, rank, pmap, fmap, batch_train, batch_test, batch_init, image_size, rnd_crop=False):
    """
    Get ImageNet data loaders for distributed training
    
    Parameters:
    data_dir (str): Directory path to dataset
    n_procs (int): Number of processes (distributed training)
    rank (int): Rank of current process
    pmap (int): Number of threads for parallel mapping
    fmap (int): Number of threads for file reading
    batch_train (int): Batch size for training
    batch_test (int): Batch size for testing
    batch_init (int): Batch size for initialization
    image_size (int): Target image size
    rnd_crop (bool): Whether to use random cropping
    
    Returns:
    train_iterator, test_iterator, init_iterator: iterators for training, testing and initialization
    """
    # Define transformations
    if rnd_crop:
        # With random crop
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        # Without random crop
        transform_train = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    # Get training and validation dataset paths
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    
    # Create datasets
    train_dataset = ImageNetDataset(train_dir, transform=transform_train)
    test_dataset = ImageNetDataset(val_dir, transform=transform_test)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=n_procs, rank=rank, shuffle=True)
    test_sampler = DistributedSampler(
        test_dataset, num_replicas=n_procs, rank=rank, shuffle=False)
    
    # Set number of workers based on input parameters
    num_workers_train = pmap
    num_workers_test = fmap
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_train,
        shuffle=False,  # Sampler handles shuffling
        num_workers=num_workers_train,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_test,
        shuffle=False,
        num_workers=num_workers_test,
        pin_memory=True,
        sampler=test_sampler,
        drop_last=True
    )
    
    # For initialization, we create a separate loader
    init_loader = DataLoader(
        train_dataset,
        batch_size=batch_init,
        shuffle=False,
        num_workers=num_workers_train,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    
    # Create iterators for TensorFlow-like usage
    def train_iterator():
        for data, target in train_loader:
            yield data, target
    
    def test_iterator():
        for data, target in test_loader:
            yield data, target
    
    def init_iterator():
        for data, target in init_loader:
            yield data, target
    
    return train_iterator, test_iterator, init_iterator


# Additional dataset classes for other datasets like LSUN, CelebA

class LSUNDataset(Dataset):
    def __init__(self, root_dir, transform=None, category=None):
        self.root_dir = root_dir
        self.transform = transform
        self.category = category
        self.imgs = []
        
        # If a specific category is requested, use only that folder
        if category:
            search_path = os.path.join(root_dir, category, '*.webp')
        else:
            search_path = os.path.join(root_dir, '*', '*.webp')
            
        self.imgs = sorted(glob.glob(search_path))
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        # LSUN is an unlabeled dataset, so we return 0 as a dummy label
        label = 0
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label


class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = sorted(glob.glob(os.path.join(root_dir, '*.jpg')))
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        # CelebA is typically used without labels for generative modeling
        label = 0
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label