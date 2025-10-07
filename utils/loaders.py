import os
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def identity(x):
    return x

def get_transforms(dataset_name, arch='resnet50'):
    """Get training and testing transforms for different datasets."""
    
    # Determine input size based on architecture
    if 'vit' in arch.lower() or 'swin' in arch.lower():
        input_size = 224  # Standard for transformers
    elif dataset_name == 'imagenet':
        input_size = 224
    elif dataset_name == 'tiny-imagenet':
        input_size = 64   # Keep native size for tiny-imagenet
    else:  # CIFAR-10, CIFAR-100, SVHN
        input_size = 32   # Native CIFAR size

    if dataset_name == 'cifar10':
        # Use consistent CIFAR-10 normalization values
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(input_size) if input_size != 32 else transforms.Lambda(identity),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.Resize(input_size) if input_size != 32 else transforms.Lambda(identity),
            transforms.ToTensor(),
            normalize
        ])
    elif dataset_name == 'cifar100':
        # Use consistent CIFAR-100 normalization values
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(input_size) if input_size != 32 else transforms.Lambda(identity),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.Resize(input_size) if input_size != 32 else transforms.Lambda(identity),
            transforms.ToTensor(),
            normalize
        ])
    elif dataset_name == 'svhn':
        normalize = transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Resize(input_size) if input_size != 32 else transforms.Lambda(identity),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.Resize(input_size) if input_size != 32 else transforms.Lambda(identity),
            transforms.ToTensor(),
            normalize
        ])
    elif dataset_name == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.Resize(input_size) if input_size != 32 else transforms.Lambda(identity),
            transforms.ToTensor(),
            normalize
        ])
    elif dataset_name == 'tiny-imagenet':
        # Use Tiny-ImageNet specific normalization values consistently
        normalize = transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        transform_train = transforms.Compose([
            transforms.Resize(input_size) if input_size != 64 else transforms.Lambda(identity),
            transforms.ToTensor(),
            normalize
        ])
        transform_test = transforms.Compose([
            transforms.Resize(input_size) if input_size != 64 else transforms.Lambda(identity),
            transforms.ToTensor(),
            normalize
        ])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return transform_train, transform_test


class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x)
        y = self.tensors[1][index]
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


def prepare_dataset(dataset_name, batch_size=None, load_train=True, num_workers=8, shuffle=False,
                    data_dir='./data', return_loader=True, arch='resnet50'):
    """Prepare the dataset and optionally return a DataLoader or just the Dataset."""
    transform_train, transform_test = get_transforms(dataset_name, arch)
    transform = transform_train if load_train else transform_test

    if dataset_name == 'cifar10':
        dataset = datasets.CIFAR10(root=data_dir, train=load_train, download=True, transform=transform)
    elif dataset_name == 'cifar100':
        dataset = datasets.CIFAR100(root=data_dir, train=load_train, download=True, transform=transform)
    elif dataset_name == 'svhn':
        split = 'train' if load_train else 'test'
        dataset = datasets.SVHN(root=data_dir, split=split, download=True, transform=transform)
    elif dataset_name == 'imagenet':
        datadir = os.path.join(data_dir, 'train' if load_train else 'test')
        dataset = datasets.ImageFolder(datadir, transform)
    elif dataset_name == 'tiny-imagenet':
        datadir = os.path.join(data_dir, 'tiny-imagenet-200', 'train' if load_train else 'test')
        dataset = datasets.ImageFolder(datadir, transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if return_loader:
        if batch_size is None:
            raise ValueError("batch_size must be provided when return_loader=True")
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                 shuffle=shuffle, num_workers=num_workers,
                                                 pin_memory=True)
        return dataloader
    else:
        return dataset


def train_val_split(args):
    """Get train and validation loaders for any dataset."""
    print(f"==> Preparing dataset {args.dataset}")
    num_workers = min(args.workers, 12)  # Fix workers issue
    
    if args.dataset == 'tiny-imagenet':
        print("Using Tiny-ImageNet validation set")
        
        # Get transforms
        transform_train, _ = get_transforms(args.dataset, args.arch)
        # Train dataset
        train_dataset = prepare_dataset(args.dataset, return_loader=False, data_dir=args.data_dir, arch=args.arch)
        
        # Validation dataset - directly load from val folder
        val_datadir = os.path.join(args.data_dir, 'tiny-imagenet-200', 'val')
        val_dataset = datasets.ImageFolder(val_datadir, transform_train)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True, 
            persistent_workers=True if num_workers > 0 else False  
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
    else:
        # CIFAR/SVHN: Split train set
        print("Splitting training set into train/val (5000 samples for validation)")
        
        dataset = prepare_dataset(args.dataset, return_loader=False, data_dir=args.data_dir, arch=args.arch)
        targets = np.array(getattr(dataset, 'targets', getattr(dataset, 'labels', None)))
        if targets is None:
            raise ValueError("Dataset must have 'targets' or 'labels'")
        
        train_idx, val_idx = train_test_split(np.arange(len(dataset)), test_size=5000, 
                                            stratify=targets, random_state=42)
        train_set, val_set = Subset(dataset, train_idx), Subset(dataset, val_idx)
        
        train_loader = DataLoader(
            train_set, 
            batch_size=args.batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True, 
            persistent_workers=True if num_workers > 0 else False  
        )
        val_loader = DataLoader(
            val_set, 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True if num_workers > 0 else False
        )
    
    return train_loader, val_loader
