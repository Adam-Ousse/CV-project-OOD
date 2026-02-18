import torch
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from torchvision import datasets, transforms
from pathlib import Path
import os
import urllib.request
import zipfile
import config

def get_cifar100_loaders():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    
    train_dataset = datasets.CIFAR100(config.DATA_DIR, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(config.DATA_DIR, train=False, download=True, transform=transform_test)
    
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)
    
    return train_loader, val_loader, test_loader

def get_ood_transform():
    return transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])

# Near-OOD
def get_cifar10_loader():
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    dataset = datasets.CIFAR10(config.DATA_DIR, train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

def get_tiny_imagenet_loader():
    data_dir = Path(config.DATA_DIR) / 'tiny-imagenet-200'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    
    # download 
    if not data_dir.exists():
        zip_path = Path(config.DATA_DIR) / 'tiny-imagenet-200.zip'
        if not zip_path.exists():
            print(f"Downloading Tiny ImageNet from {url}...")
            os.makedirs(config.DATA_DIR, exist_ok=True)
            urllib.request.urlretrieve(url, zip_path)
        print(f"Extracting to {data_dir}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(Path(config.DATA_DIR))
    transform = get_ood_transform()
    test_dir = data_dir / 'test'
    dataset = datasets.ImageFolder(str(test_dir), transform=transform)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

# Far-OOD 
def get_mnist_loader():
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST(config.DATA_DIR, train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

def get_svhn_loader():
    transform = get_ood_transform()
    dataset = datasets.SVHN(config.DATA_DIR, split='test', download=True, transform=transform)
    # first 10k samples
    dataset = Subset(dataset, range(min(10000, len(dataset))))
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

def get_textures_loader():
    transform = get_ood_transform()
    dataset = datasets.DTD(config.DATA_DIR, split='test', download=True, transform=transform)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

def get_ood_loader(ood_type='all'):
    if ood_type == 'near':
        return {
            'cifar10': get_cifar10_loader(),
            'tiny_imagenet': get_tiny_imagenet_loader(),
        }
    elif ood_type == 'far':
        return {
            'mnist': get_mnist_loader(),
            'svhn': get_svhn_loader(),
            'textures': get_textures_loader(),
        }
    elif ood_type =="all":
        return {
            'cifar10': get_cifar10_loader(),
            'tiny_imagenet': get_tiny_imagenet_loader(),
            'mnist': get_mnist_loader(),
            'svhn': get_svhn_loader(),
            'textures': get_textures_loader(),
        }
    else:
        raise ValueError(f"unknown ood_type: {ood_type}. Use 'near', 'far', or specific dataset name")