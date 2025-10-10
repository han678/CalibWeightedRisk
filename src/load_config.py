import os
import yaml
import torch.nn as nn
from loss.aurc import AURCLoss
from loss.dual_focal import DualFocalLoss
from loss.ece_kde import ECE_KDELoss
from loss.focal import FocalLoss
from loss.inverse_focal import InverseFocalLoss
from loss.selectAU import SelectiveAULoss
import argparse
def create_parser(description='Training a selective classifier'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--arch', '-a', default='resnet50',
                        help='Model architecture (default: resnet50)')
    parser.add_argument('-d', '--dataset', default='cifar10',
                        choices=['cifar10', 'svhn', 'cifar100', 'tiny-imagenet', 'imagenet'])
    parser.add_argument('-j', '--workers', default=1, type=int)
    parser.add_argument('--loss_type', default="aurc", type=str,
                        choices=['aurc', "focal", "inverse_focal", "dual_focal", "ece_kde", "ce", "select_au"])
    parser.add_argument('--score_function', default="NegEntropy", type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training and testing')
    parser.add_argument('--opt', default="sgd", help='Optimizer type')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--milestones', default=[60, 120, 160], type=int, nargs='+',
                        help='Learning rate scheduler milestones')
    parser.add_argument('--lr_gamma', default=0.2, type=float,
                        help='Learning rate decay gamma (different from loss gamma)')
    parser.add_argument('--scheduler', default='multistep', type=str,
                        choices=['multistep', 'cosine'],
                        help='Learning rate scheduler type')
    parser.add_argument('--load_pretrain', default=0, type=int)
    parser.add_argument('-e', '--evaluate', default=False, type=bool)
    parser.add_argument('--model_dir', default='./result/pretrain', type=str)
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--seed', default=40, type=int)
    parser.add_argument('--gamma', type=float, default=None, help='Gamma value for focal/inverse_focal/dual_focal')
    parser.add_argument('--conf_quantile', type=float, default=0.25, help='Confidence quantile for selectAU loss')
    parser.add_argument('--hist_sigma', type=float, default=0.05, help='Histogram sigma for selectAU loss')
    parser.add_argument('--use_amp', default=True, type=bool, help='Use automatic mixed precision training (default: True)')
    parser.add_argument('--use_augmentation', default=True, type=bool, help='Use data augmentation (default: True)')
    return parser
    
def load_yaml_config(dataset, arch):
    config_path = f"config/{dataset}/{arch}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def get_loss_params(cfg):
    loss_type = cfg['loss_type']
    loss_params = cfg['loss_configs'][loss_type]
    return loss_type, loss_params

def get_loss_function(loss_type, loss_params):
    loss_map = {
        "ce": lambda p: nn.CrossEntropyLoss(reduction=p.get('reduction', 'mean')),
        "aurc": lambda p: AURCLoss(**p),
        "ece_kde": lambda p: ECE_KDELoss(**p),
        "focal": lambda p: FocalLoss(**p),
        "inverse_focal": lambda p: InverseFocalLoss(**p),
        "dual_focal": lambda p: DualFocalLoss(**p),
        "select_au": lambda p: SelectiveAULoss(**p)
    }
    if loss_type not in loss_map:
        raise ValueError(f"Invalid loss type: {loss_type}")
    loss = loss_map[loss_type](loss_params)
    gamma = getattr(loss, 'gamma', None)
    print(f"Loss: {loss_type}, gamma: {gamma}")
    return loss

def get_save_criteria(dataset, total_epochs):
    warmup_epochs = {
        'cifar10': min(40, total_epochs // 3), 
        'cifar100': min(40, total_epochs // 3), 
        'tiny-imagenet': min(20, total_epochs // 3)
    }
    return {
        'warmup_epoch': warmup_epochs.get(dataset, max(60, total_epochs // 3))
    }

def get_model_names():
    return [
        "VGG16Drop", "VGG16BNDrop", "VGG16", "VGG16BN",
        "VGG19Drop", "VGG19BNDrop", "VGG19", "VGG19BN",
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet110',
        'resnet152', 'wrn', 'PreResNet20', 'PreResNet20Drop', 'PreResNet56',
        'PreResNet56Drop', 'PreResNet110', 'PreResNet110Drop', 'PreResNet164',
        'PreResNet164Drop', 'PreResNet', 'vit_small', 'vit_base', 'swin_tiny', 'swin_base'
    ]
