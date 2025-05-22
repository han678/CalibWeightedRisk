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
    return ['resnet50',  'resnet110', 'wrn', 'PreResNet56']

def common_args():
    model_names = get_model_names()
    """Setup and parse common command line arguments."""
    parser = argparse.ArgumentParser(description='Training a selective classifier')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: VGG16)')
    parser.add_argument('-d', '--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tiny-imagenet'])
    parser.add_argument('-j', '--workers', default=16, type=int)
    parser.add_argument('--loss_type', default='aurc', type=str,
                        choices=['aurc', "focal", "inverse_focal", "dual_focal", "ece_kde", "ce"])
    parser.add_argument('--score_function', default="NegEntropy", type=str)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--train-batch', default=128, type=int)
    parser.add_argument('--test-batch', default=128, type=int)
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, metavar='W',
                        help='weight decay (default: 5e-4)')
    parser.add_argument('--load_pretrain', default=0, type=int)
    parser.add_argument('-e', '--evaluate', default=False, type=bool)
    parser.add_argument('--model_dir', default='./result/pretrain', type=str,
                        help='path to the folder that contains pretrained model')
    parser.add_argument('--data_dir', default='./data', type=str)
    parser.add_argument('--seed', default=10, type=int)
    parser.add_argument('--gamma', default=0.5, type=float, help='Gamma parameter for loss')  # Add additional argument
    parser.add_argument('--weight_grad', type=str2bool, default=False, help='whether to consider weight gradient for AURC loss')  # Add additional argument
    parser.add_argument('--regularization_strength', default=0.05, type=float, help='Regularization strength for AURC loss')  # Add additional argument
    return parser

def get_loss_function_and_name(args, loss_type, score_function, input_is_softmax=False, regularization_strength=0.05):
    if loss_type == "ce":
        gamma = None
        loss = nn.CrossEntropyLoss(reduction='mean')
        loss_name = "CrossEntropy"
    elif loss_type == "aurc":
        gamma = args.gamma
        weight_grad = args.weight_grad
        print(f"loss: {loss_type}, gamma:{gamma}, weight_grad:{weight_grad}, regu:{regularization_strength}")
        loss = AURCLoss(batch_size=args.train_batch, score_function=score_function, gamma=gamma, reduction='mean', 
                        weight_grad=weight_grad, input_is_softmax=input_is_softmax, regularization_strength=regularization_strength)
        loss_name = "AURC"
    elif loss_type == "ece_kde":
        gamma = 0.3 if args.dataset == 'cifar10' else 0.5 if args.dataset == 'cifar100' else 0.3
        print(f"loss: {loss_type}, gamma:{gamma}")
        loss = ECE_KDELoss(p=1, mc_type='canonical', gamma=gamma, input_is_softmax=input_is_softmax)
        loss_name = "ECE-KDE"
    elif loss_type == "focal":
        gamma = None
        loss = FocalLoss(gamma=gamma, size_average=True, input_is_softmax=input_is_softmax)
        loss_name = f"Focal $\gamma={gamma}$" if gamma is not None else "FL-53"
    elif loss_type == "inverse_focal":
        gamma = get_gamma_inverse_focal(dataset=args.dataset, arch=args.arch)
        loss = InverseFocalLoss(gamma=gamma, size_average=True, input_is_softmax=input_is_softmax)
        loss_name = f"InverseFocal $\gamma={gamma}$"
    elif loss_type == "dual_focal":
        gamma = get_gamma_dual_focal(dataset=args.dataset, arch=args.arch)
        print(f"loss: {loss_type}, gamma:{gamma}")
        loss = DualFocalLoss(gamma=gamma, size_average=True, input_is_softmax=input_is_softmax)
        loss_name = f"DualFocal $\gamma={gamma}$ "
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
    print(f"loss: {loss_type}, gamma:{gamma}")
    return loss, loss_name


def get_gamma_dual_focal(dataset, arch):
    """
    Returns the gamma value for the dual focal loss based on the dataset and model architecture.
    """
    gamma_values = {
        "cifar100": {
            "resnet50": 5,
            "resnet110": 6.1,
            "wrn": 3.9,
            "densenet121": 3.4,
            'PreResNet56': 4.5,
        },
        "cifar10": {
            "resnet50": 2.0,
            "resnet110": 4.5,
            "wrn": 2.6,
            "densenet121": 5,
            'PreResNet56': 4.0,
        },
        "tiny-imagenet": {
            "resnet50": 2.3,
        },
    }
    try:
        return gamma_values[dataset][arch]
    except KeyError:
        raise ValueError(f"Gamma value not defined for dataset '{dataset}' and architecture '{arch}'")

def get_gamma_inverse_focal(dataset, arch):
    """
    Returns the gamma value for the inverse focal loss based on the dataset and model architecture.
    """
    gamma_values = {
        "cifar10": {
            "resnet50": 2.0,
            "resnet110": 3.0,
            "wrn": 1.5,
            'PreResNet56': 2.0,
            "densenet121": 2.0,
        },
        "cifar100": {
            "resnet50": 1.0,
            "resnet110": 1.5,
            "wrn": 1.0,
            'PreResNet56': 1.0,
            "densenet121": 1.5,
        },
        "tiny-imagenet": {
            "resnet50": 1.0,
        },
    }
    try:
        return gamma_values[dataset][arch]
    except KeyError:
        raise ValueError(f"Gamma value not defined for dataset '{dataset}' and architecture '{arch}'")
    
    

    

    
