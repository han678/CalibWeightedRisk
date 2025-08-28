import argparse
import torch.nn as nn
from loss.aurc import AURCLoss
from loss.dual_focal import DualFocalLoss
from loss.ece_kde import ECE_KDELoss
from loss.focal import FocalLoss
from loss.inverse_focal import InverseFocalLoss


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ('true', '1', 'yes')

def get_model_names():
    """Returns a list of supported model names."""
    return [
        "VGG16Drop", "VGG16BNDrop", "VGG16", "VGG16BN",
        "VGG19Drop", "VGG19BNDrop", "VGG19", "VGG19BN",
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet110',
        'resnet152', 'wrn', 'PreResNet20', 'PreResNet20Drop', 'PreResNet56',
        'PreResNet56Drop', 'PreResNet110', 'PreResNet110Drop', 'PreResNet164',
        'PreResNet164Drop', 'PreResNet', 'densenet121', 'densenet169',
        'vit_small', 'vit_base', 'swin_tiny', 'swin_base'
    ]

def get_training_config(dataset, arch='resnet50'):
    """Get training configuration based on dataset following paper specifications."""
    other_config = {
        'batch_size': 256, 'epochs': 90, 'lr': 1e-4, 'optimizer': 'sgd',
        'scheduler': 'cosine', 'weight_decay': 5e-4, 'momentum': 0.9,
        'milestones': [], 'lr_gamma': 0.2
    }
    
    if arch not in ['vit_small', 'vit_base', 'swin_tiny', 'swin_base']:
        # CNN configurations
        configs = {
            'cifar10': {
                'batch_size': 128, 'epochs': 200, 'lr': 0.1, 'weight_decay': 5e-4,
                'momentum': 0.9, 'milestones': [60, 120, 160], 'lr_gamma': 0.2, 'optimizer': 'sgd',
                'scheduler': 'multistep'
            },
            'cifar100': {
                'batch_size': 128, 'epochs': 200, 'lr': 0.1, 'weight_decay': 5e-4,
                'momentum': 0.9, 'milestones': [60, 120, 160], 'lr_gamma': 0.2, 'optimizer': 'sgd',
                'scheduler': 'multistep'
            },
            'tiny-imagenet': {
                'batch_size': 256, 'epochs': 90, 'lr': 0.1, 'weight_decay': 1e-4,
                'momentum': 0.9, 'milestones': [30, 60], 'lr_gamma': 0.1, 'optimizer': 'sgd',
                'scheduler': 'multistep'
            },
            'imagenet': {
                'batch_size': 256, 'epochs': 100, 'lr': 0.01, 'weight_decay': 1e-3,
                'momentum': 0.9, 'milestones': [], 'lr_gamma': 1.0, 'optimizer': 'sgd',
                'scheduler': 'cosine'
            }
        }
        return configs.get(dataset, other_config)
    else:
        # For ViT and Swin transformers, use the other_config
        return other_config

def create_parser(description='Training a selective classifier', default_dataset='cifar10'):
    """Create a common argument parser with default values."""
    parser = argparse.ArgumentParser(description=description)
    model_names = get_model_names()
    
    # Model and dataset arguments
    parser.add_argument('--arch', '-a', default='resnet50', choices=model_names,
                        help='Model architecture (default: resnet50)')
    parser.add_argument('-d', '--dataset', default=default_dataset, 
                        choices=['cifar10', 'svhn', 'cifar100', 'tiny-imagenet', 'imagenet'])
    parser.add_argument('-j', '--workers', default=2, type=int)
    
    # Training arguments
    parser.add_argument('--loss_type', default='focal', type=str,
                        choices=['aurc', "focal", "inverse_focal", "dual_focal", "ece_kde", "ce"])
    parser.add_argument('--score_function', default="NegEntropy", type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training and testing')
    parser.add_argument('--opt', default="sgd", help='Optimizer type')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    
    # Additional training parameters from get_training_config
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--milestones', default=[60, 120, 160], type=int, nargs='+', 
                        help='Learning rate scheduler milestones')
    parser.add_argument('--lr_gamma', default=0.2, type=float, 
                        help='Learning rate decay gamma (different from loss gamma)')
    parser.add_argument('--scheduler', default='multistep', type=str, 
                        choices=['multistep', 'cosine'], 
                        help='Learning rate scheduler type')
    
    # Model loading and paths
    parser.add_argument('--load_pretrain', default=0, type=int) # only change when applying post hoc calibration
    parser.add_argument('-e', '--evaluate', default=False, type=bool)
    parser.add_argument('--model_dir', default='./result/pretrain', type=str)
    parser.add_argument('--data_dir', default='./data', type=str)
    
    # Loss-specific parameters
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--gamma', type=float, default=None, help='Gamma value for focal/inverse_focal/dual_focal/aurc')
    parser.add_argument('--weight_grad', type=str2bool, default=True, 
                        help='Whether to consider weight gradient for AURC loss')
    parser.add_argument('--regularization_strength', default=0.05, type=float, 
                        help='Regularization strength for AURC loss')
    
    # Training optimizations
    parser.add_argument('--use_amp', type=str2bool, default=True, 
                        help='Use automatic mixed precision training (default: True)')
    parser.add_argument('--use_augmentation', type=str2bool, default=True, 
                        help='Use data augmentation (default: True)')
    
    return parser

def configure_training_params(args):
    """Configure training parameters and loss-specific parameters based on dataset and architecture."""
    config = get_training_config(args.dataset, args.arch)  # Pass architecture

    # Training configuration
    args.batch_size = config['batch_size']
    args.epochs = config['epochs']
    args.lr = config['lr']
    args.weight_decay = config.get('weight_decay', 5e-4)
    args.momentum = config.get('momentum', 0.9)
    args.milestones = config.get('milestones', [])
    args.lr_gamma = config.get('lr_gamma', 0.2)
    args.opt = config['optimizer']
    args.scheduler = config.get('scheduler', 'cosine')
    args._training_config = config

    # Loss-specific configuration
    gamma_table = get_gamma_values()
    if args.loss_type in gamma_table and args.gamma is None:
        args.gamma = gamma_table[args.loss_type].get(args.dataset, {}).get(args.arch, 0.5)
    args.regularization_strength = get_regularization_strength(args.dataset)
    # Print configuration for logging/debugging
    print(f"configuration for {args.dataset} with {args.arch}:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Momentum: {args.momentum}")
    print(f"  Milestones: {args.milestones}")
    print(f"  LR gamma: {args.lr_gamma}")
    print(f"  gamma: {getattr(args, 'gamma', None)}")
    print(f"  Optimizer: {args.opt}")
    print(f"  Scheduler: {args.scheduler}")

    return args


def configure_posthoc_params(args):
    """Configure training parameters and loss-specific parameters based on dataset and architecture."""
    config = get_training_config(args.dataset, args.arch)  # Pass architecture

    # Training configuration
    args.batch_size = config['batch_size']
    args.epochs = config['epochs']
    args.lr = config['lr']
    args.weight_decay = config.get('weight_decay', 5e-4)
    args.momentum = config.get('momentum', 0.9)
    args.milestones = config.get('milestones', [])
    args.lr_gamma = config.get('lr_gamma', 0.2)
    args.opt = config['optimizer']
    args.scheduler = config.get('scheduler', 'cosine')
    args._training_config = config
    args.load_pretrain = 1  # Always load pre-trained model for post-hoc calibration
    # Loss-specific configuration
    gamma_table = get_gamma_values()
    if args.loss_type in gamma_table and args.gamma is None:
        args.gamma = gamma_table[args.loss_type].get(args.dataset, {}).get(args.arch, 0.5)
    args.regularization_strength = get_regularization_strength(args.dataset)
    print(f"configuration for {args.dataset} with {args.arch}:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Momentum: {args.momentum}")
    print(f"  Milestones: {args.milestones}")
    print(f"  LR gamma: {args.lr_gamma}")
    print(f"  gamma: {getattr(args, 'gamma', None)}")
    print(f"  Optimizer: {args.opt}")
    print(f"  Scheduler: {args.scheduler}")

    return args

def configure_search_params(args):
    """Configure search parameters based on dataset and architecture."""
    config = get_training_config(args.dataset, args.arch)  # Pass architecture

    # Training configuration
    args.batch_size = config['batch_size']
    args.epochs = config['epochs']
    args.lr = config['lr']
    args.weight_decay = config.get('weight_decay', 5e-4)
    args.momentum = config.get('momentum', 0.9)
    args.milestones = config.get('milestones', [])
    args.lr_gamma = config.get('lr_gamma', 0.2)
    args.opt = config['optimizer']
    args.scheduler = config.get('scheduler', 'cosine')
    args._training_config = config

    # Loss-specific configuration
    args.regularization_strength = get_regularization_strength(args.dataset)
    # Print configuration for logging/debugging
    print(f"configuration for {args.dataset} with {args.arch}:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Momentum: {args.momentum}")
    print(f"  Milestones: {args.milestones}")
    print(f"  LR gamma: {args.lr_gamma}")
    print(f"  gamma: {getattr(args, 'gamma', None)}")
    print(f"  Optimizer: {args.opt}")
    print(f"  Scheduler: {args.scheduler}")

    return args

def get_gamma_values():
    """Return gamma values for different loss types, datasets, and architectures."""
    return {
        "dual_focal": {
            "cifar100": {"resnet50": 5, "resnet110": 6.1, "wrn": 3.9, "densenet121": 3.4, 'PreResNet56': 4.5},
            "cifar10": {"resnet50": 2.0, "resnet110": 4.5, "wrn": 2.6, "densenet121": 5, 'PreResNet56': 4.0},
            "tiny-imagenet": {"resnet50": 2.3, "vit_small": 6.0, "swin_tiny": 6.0}
        },
        "inverse_focal": {
            "cifar10": {"resnet50": 2.0, "resnet110": 3.0, "wrn": 1.5, 'PreResNet56': 2.0, "densenet121": 2.0},
            "cifar100": {"resnet50": 1.0, "resnet110": 1.5, "wrn": 1.0, 'PreResNet56': 1.0, "densenet121": 1.5},
            "tiny-imagenet": {"resnet50": 1.0, "vit_small": 1.0, "swin_tiny": 1.0}
        },
        "aurc": {
            "cifar10": {
                "resnet50": 0.9,
                "resnet110": 0.9,
                "wrn": 0.9,
                "PreResNet56": 0.9
            },
            "cifar100": {
                "resnet50": 0.5,
                "resnet110": 0.9,
                "wrn": 0.2,
                "PreResNet56": 0.8
            },
            "tiny-imagenet": {
                "vit_small": 0.8,
                "swin_tiny": 0.9
            }
        },
        "ece_kde": {
            "cifar10": {"resnet50": 0.3, "resnet110": 0.3, "wrn": 0.3, "densenet121": 0.3, 'PreResNet56': 0.3},
            "cifar100": {"resnet50": 0.5, "resnet110": 0.5, "wrn": 0.5, "densenet121": 0.5, 'PreResNet56': 0.5},
            "tiny-imagenet": {"resnet50": 0.3}
        },
    }


def get_loss_function(args, loss_type, score_function, is_prob=False, regularization_strength=0.05):
    """Create loss function based on type and parameters."""
    loss_configs = {
        "ce": lambda: nn.CrossEntropyLoss(reduction='mean'),
        "aurc": lambda: AURCLoss(
            batch_size=args.batch_size,
            score_function=score_function, 
            gamma=args.gamma,
            reduction='mean', 
            weight_grad=args.weight_grad, 
            is_prob=is_prob,
            regularization_strength=regularization_strength
        ),
        "ece_kde": lambda: ECE_KDELoss(
            p=1, 
            mc_type='canonical', 
            gamma=args.gamma,
            is_prob=is_prob
        ),
        "focal": lambda: FocalLoss(gamma=None, size_average=True, is_prob=is_prob),
        "inverse_focal": lambda: InverseFocalLoss(
            gamma=args.gamma,
            size_average=True, 
            is_prob=is_prob
        ),
        "dual_focal": lambda: DualFocalLoss(
            gamma=args.gamma,
            size_average=True, 
            is_prob=is_prob
        )
    }
    
    if loss_type not in loss_configs:
        raise ValueError(f"Invalid loss type: {loss_type}")
    
    loss = loss_configs[loss_type]()
    gamma = getattr(loss, 'gamma', None)
    print(f"Loss: {loss_type}, gamma: {gamma}")
    return loss


def get_save_criteria(dataset, total_epochs):
    """Get model saving criteria based on dataset."""
    warmup_epochs = {
        'cifar10': min(40, total_epochs // 3), 
        'cifar100': min(40, total_epochs // 3), 
        'tiny-imagenet': min(30, total_epochs // 3), 
        'imagenet': min(30, total_epochs // 3)
    }
    
    return {
        'warmup_epoch': warmup_epochs.get(dataset, max(60, total_epochs // 3))
    }

def get_regularization_strength(dataset):
    """Get regularization strength based on dataset."""
    rs_map = {"tiny-imagenet": 0.05, "cifar100": 0.01, "cifar10": 0.05}
    return rs_map.get(dataset, 0.05)





