import os
import sys
import torch.nn as nn

class SwinWithPool(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
    def forward(self, x):
        out = self.base(x)
        if out.dim() == 4:
            if out.shape[-1] == 200:
                out = out.mean(dim=(1, 2))
            elif out.shape[1] == 200:
                out = out.mean(dim=(2, 3))
        return out


def get_network(args):
    """Return given network for any supported dataset"""
    
    # Determine number of classes based on dataset
    num_classes = {
        'cifar10': 10, 
        'cifar100': 100, 
        'svhn': 10, 
        'imagenet': 1000, 
        'tiny-imagenet': 200
    }.get(args.dataset, 10)
    
    # Handle timm models (Vision Transformers and Swin)
    if args.arch.lower() in ["vit_small", "vit_base", 'swin_tiny', "swin_base"]:
        import timm
        model_names = {
            'vit_small': 'vit_small_patch16_224',
            'vit_base': 'vit_base_patch16_224', 
            'swin_tiny': 'swin_tiny_patch4_window7_224',
            'swin_base': 'swin_base_patch4_window7_224'
        }
        if args.arch.lower() in ['swin_tiny', 'swin_base']:
            net = timm.create_model(model_names[args.arch.lower()], pretrained=True, num_classes=num_classes)
            net.head = nn.Linear(net.head.in_features, num_classes)
            net = SwinWithPool(net)
            return net
        net = timm.create_model(model_names[args.arch.lower()], pretrained=True, num_classes=num_classes)
        net.head = nn.Linear(net.head.in_features, num_classes)
        return net
    elif args.arch == 'VGG16BN':
        from models.vgg import VGG16BN
        net = VGG16BN(num_classes=num_classes)
    elif args.arch == 'VGG16':
        from models.vgg import VGG16
        net = VGG16(num_classes=num_classes)
    elif args.arch == 'VGG16Drop':
        from models.vgg import VGG16Drop
        net = VGG16Drop(num_classes=num_classes)
    elif args.arch == 'VGG16BNDrop':
        from models.vgg import VGG16BNDrop
        net = VGG16BNDrop(num_classes=num_classes)
    elif args.arch == 'VGG19BN':
        from models.vgg import VGG19BN
        net = VGG19BN(num_classes=num_classes)
    elif args.arch == 'VGG19':
        from models.vgg import VGG19
        net = VGG19(num_classes=num_classes)
    elif args.arch == 'VGG19Drop':
        from models.vgg import VGG19Drop
        net = VGG19Drop(num_classes=num_classes)
    elif args.arch == 'VGG19BNDrop':
        from models.vgg import VGG19BNDrop
        net = VGG19BNDrop(num_classes=num_classes)
    elif args.arch == 'resnet18':
        from models.resnet import resnet18
        net = resnet18(num_classes=num_classes)
    elif args.arch == 'resnet34':
        from models.resnet import resnet34
        net = resnet34(num_classes=num_classes)
    elif args.arch == 'resnet50':
        from models.resnet import resnet50
        net = resnet50(num_classes=num_classes)
    elif args.arch == 'resnet101':
        from models.resnet import resnet101
        net = resnet101(num_classes=num_classes)
    elif args.arch == 'resnet110':
        from models.resnet import resnet110
        net = resnet110(num_classes=num_classes)
    elif args.arch == 'resnet152':
        from models.resnet import resnet152
        net = resnet152(num_classes=num_classes)
    elif args.arch == 'wrn':
        from models.wrn import WideResNet28x10
        net = WideResNet28x10(num_classes=num_classes)
    elif args.arch == 'PreResNet20':    
        from models.preresnet import PreResNet20
        net = PreResNet20(num_classes=num_classes)
    elif args.arch == 'PreResNet20Drop':
        from models.preresnet import PreResNet20Drop
        net = PreResNet20Drop(num_classes=num_classes)
    elif args.arch == 'PreResNet56':
        from models.preresnet import PreResNet56
        net = PreResNet56(num_classes=num_classes)
    elif args.arch == 'PreResNet56drop':
        from models.preresnet import PreResNet56Drop
        net = PreResNet56Drop(num_classes=num_classes)
    elif args.arch == 'PreResNet110':
        from models.preresnet import PreResNet110
        net = PreResNet110(num_classes=num_classes)
    elif args.arch == 'PreResNet110drop':
        from models.preresnet import PreResNet110Drop
        net = PreResNet110Drop(num_classes=num_classes)
    elif args.arch == 'PreResNet164':
        from models.preresnet import PreResNet164
        net = PreResNet164(num_classes=num_classes)
    elif args.arch == 'PreResNet164drop':
        from models.preresnet import PreResNet164Drop
        net = PreResNet164Drop(num_classes=num_classes)
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    return net