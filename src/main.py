from __future__ import print_function
import random
import numpy as np
import os
import torch.backends.cudnn as cudnn, torch.nn as nn

from common import common_args, get_loss_function_and_name
from loss.ece import Ece
from models.utils import get_network
from utils import Logger
import torch
import torch.nn.functional as F
import timm
from loss.ece_kde import get_ece_kde
from utils.aurc_estimators import get_em_AURC
from utils.loaders import prepare_dataset
from loss.aurc import cross_entropy_loss, get_score_function
from utils.metrics import get_brier_score
from posthoc.ts import ModelWithTemperature

def train(net, train_loader, criterion, optimizer):
    """Train the network for one epoch."""
    net.train()
    train_loss = 0
    total_correct_1 = 0
    total_correct_5 = 0
    total = 0
    device = next(net.parameters()).device
    for batch_idx, (images, targets) in enumerate(train_loader):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            logits = F.softmax(outputs, dim=1)
            _, pred = logits.topk(5, 1, largest=True, sorted=True)
            label = targets.view(targets.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()
        train_loss += loss.data.item()
        total_correct_1 += correct[:, :1].sum()
        total_correct_5 += correct[:, :5].sum()
        total += targets.size(0)
        top1_acc = 100. * float(total_correct_1) / float(total)
        top5_acc = 100. * float(total_correct_5) / float(total)
        avg_loss = train_loss / (batch_idx + 1)
        if batch_idx % 50 == 0:
            print('Train Set: {}, {} Loss: {:.2f} | Acc: {:.2f}% ({}/{}) '.format(
                batch_idx, len(train_loader), avg_loss, top1_acc, total_correct_1, total))
    return avg_loss, top1_acc, top5_acc


def evaluate(model, val_loader, score_func="MSP", temp_scale=False, temp_criterion=nn.CrossEntropyLoss()):
    model.eval()
    all_probs, all_logits, all_targets, all_confidences = [], [], [], []
    total_correct_1 = 0
    total_correct_5 = 0
    device = next(model.parameters()).device
    score_func = get_score_function(score_func)
    eps = 1e-9
    if temp_scale:
        ts_model = ModelWithTemperature(model)
        ts_model.set_temp(val_loader, temp_criterion)
    with torch.no_grad():
        for _, (images, targets) in enumerate(val_loader):
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            logits = logits if not temp_scale else ts_model.temp_scale(logits)
            probs = torch.clamp(F.softmax(logits, dim=1), min=eps, max=1 - eps)
            confidence = score_func(logits)
            _, pred = probs.topk(5, 1, largest=True, sorted=True)
            correct = pred.eq(targets.view(targets.size(0), -1).expand_as(pred)).float()
            total_correct_1 += correct[:, :1].sum().item()
            total_correct_5 += correct[:, :5].sum().item()

            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_targets.append(F.one_hot(targets, num_classes=probs.shape[1]).to(device).cpu().numpy())
            all_confidences.append(confidence.cpu().numpy())
    all_logits = np.vstack(all_logits)
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    all_confidences = np.hstack(all_confidences)
    loss = cross_entropy_loss(all_probs, all_targets)
    top1_acc = 100. * total_correct_1 / len(val_loader.dataset)
    top5_acc = 100. * total_correct_5 / len(val_loader.dataset)

    ece = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=False, classwise=False, is_input_softmax=False)
    ece_adaptive = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=True, classwise=False, is_input_softmax=False)
    cwece = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=False, classwise=True, is_input_softmax=False)
    cwece_adaptive = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=True, classwise=True, is_input_softmax=False)
    cwece_adaptive.set_bins(torch.from_numpy(all_confidences))
    ece_adaptive.set_bins(torch.from_numpy(all_confidences))

    bandwidth = 0.02 #get_bandwidth(all_probs, device='cpu')
    result = {
        "acc_1": top1_acc,
        "acc_5": top5_acc,
        "l1_ce": get_ece_kde(all_probs, all_targets, bandwidth=bandwidth, p=1, mc_type='canonical', device='cpu') * 100.0,
        "l2_ce": get_ece_kde(all_probs, all_targets, bandwidth=bandwidth, p=2, mc_type='canonical', device='cpu') * 100.0,
        "brier_score": get_brier_score(all_probs, all_targets) * 100.0,
        "ece": ece(logits=torch.from_numpy(all_logits), labels=torch.from_numpy(all_targets)).item() * 100.0,
        "ece_a":  ece_adaptive(logits=torch.from_numpy(all_logits), labels=torch.from_numpy(all_targets)).item() * 100.0,
        "cwece": cwece(logits=torch.from_numpy(all_logits), labels=torch.from_numpy(all_targets)) * 100.0,
        "cwece_a": cwece_adaptive(logits=torch.from_numpy(all_logits), labels=torch.from_numpy(all_targets)) * 100.0,
        "aurc": get_em_AURC(residuals=loss, confidence=all_confidences),
    }
    return result


def export_logits(model, data_loader, save_name, device):
    model.eval()
    logits = []
    targets = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            logits.append(model(data).cpu().numpy())
            targets.append(target.numpy())
    logits = np.vstack(logits)
    targets = np.hstack(targets)
    np.savez(save_name, logits=logits, targets=targets)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print('==> Preparing dataset %s' % args.dataset)
    train_loader = prepare_dataset(
        args.dataset, args.train_batch, True, args.workers, args.data_dir
    )
    val_loader = prepare_dataset(
        args.dataset, args.train_batch, False, args.workers, args.data_dir
    )
    root = "train_output/"
    if args.loss_type == "aurc" and args.weight_grad == False:
        save_model_path = root + f"models/{args.dataset}/{args.score_function}{str(args.weight_grad)}"
        save_npz_path = root + f"npz/{args.dataset}/{args.score_function}{str(args.weight_grad)}/"
        output_path = root + args.dataset + "/seed" + str(args.seed) + args.score_function + str(args.weight_grad)
    elif args.loss_type == "aurc" and args.weight_grad == True:
        save_model_path = root + f"models/{args.dataset}/{args.score_function}"
        save_npz_path = root + f"npz/{args.dataset}/{args.score_function}/"
        output_path = root + args.dataset + "/seed" + str(args.seed) + args.score_function 
    else:
        save_model_path = root + f"models/{args.dataset}/"
        save_npz_path = root + f"npz/{args.dataset}/"
        output_path = root + args.dataset + "/seed" + str(args.seed)
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(save_model_path, exist_ok=True)
        os.makedirs(save_npz_path, exist_ok=True)

    if args.dataset != "tiny-imagenet":
        model = get_network(args)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model.cuda())
            cudnn.benchmark = True
        if args.load_pretrain:
            print("==> Load pretrained model '%s'" % args.arch)
            model_path = args.model_dir + "/" + args.dataset + "/" + args.arch + f"_seed25.pth"
            model.load_state_dict(torch.load(model_path))
            if torch.cuda.is_available():
                model = torch.nn.DataParallel(model.cuda())
                cudnn.benchmark = True
        else:
            print("==> Randomly initialize the model")
    else:
        model = timm.create_model(args.arch.lower(), pretrained=True, num_classes=200)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model.cuda())
            cudnn.benchmark = True
    score_function = args.score_function
    loss_type = args.loss_type
    regularization_strength = args.regularization_strength
    loss, loss_name = get_loss_function_and_name(args, loss_type, score_function, input_is_softmax=False, regularization_strength=regularization_strength)
    results = evaluate(model, val_loader, score_func="MSP")
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    if args.dataset == 'tiny-imagenet':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60], gamma=0.1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

    title = f"{args.dataset}-{loss_type}-{args.arch}"
    # set up the output format
    loss_name = args.loss_type if args.loss_type != "aurc" else args.loss_type + f"_{args.gamma}_rs{regularization_strength}"
    logger = Logger(os.path.join(output_path, f'{args.arch}_{loss_name}.txt'), title=title)
    add_names = ["Train Acc1", "Train Acc5", "Val Acc1", "Val Acc5"] if args.dataset in ['cifar100', 'imagenet'] else ['Train Acc', "Val Acc"]
    logger.set_names(['Epoch', 'Train Loss'] + add_names + ["l1CE", "l2CE", 'Brier', "ece", "ece_a", "cwece", "cwece_a", 'AURC'])
    model_info = [0, None, None, None] + list(results.values()) if args.dataset in ['cifar100', 'imagenet'] else [0,None,None] + list(
        results.values())[:1] + list(results.values())[2:]
    logger.append(model_info)
    export_logits(model, train_loader, f'{save_npz_path}/{args.arch}_init_seed{args.seed}_train_logits', device)
    export_logits(model, val_loader, f'{save_npz_path}/{args.arch}_init_seed{args.seed}_val_logits', device)
    # Initialize best ECE
    best_ece = float('inf')
    best_model_path = f"{save_model_path}/{args.arch}_{loss_name}_best_seed{args.seed}.pth"
    # start training
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train(model, train_loader, loss, optimizer)
        print("lr: ", optimizer.param_groups[0]['lr'])
        lr_scheduler.step()
        results = evaluate(model, val_loader, score_func="MSP")
        add_res = [train_acc_5] + list(results.values()) if args.dataset in ['cifar100', 'imagenet'] else list(
            results.values())[:1] + list(results.values())[2:]  # remove the top 5 accuracy
        logger.append([int(epoch), train_loss, train_acc_1] + add_res)
        current_ece = results["ece"]
        min_epoch = 120
        acc_threshold = 88.0 if args.dataset == "cifar10" else 60.0 if args.dataset == "cifar100" else 74.0
        if current_ece < best_ece and epoch >= min_epoch and train_acc_5 > acc_threshold:
            best_ece = current_ece
            torch.save(model.state_dict(), best_model_path)
            export_logits(model, train_loader, f'{save_npz_path}/{args.arch}_{loss_name}_best_seed{args.seed}_train_logits', device)
            export_logits(model, val_loader, f'{save_npz_path}/{args.arch}_{loss_name}_best_seed{args.seed}_val_logits', device)
            print(f"Epoch {epoch}: New best ECE {best_ece:.4f}. Model saved to {best_model_path}.")
    
    logger.close()


if __name__ == '__main__':
    parser = common_args()
    args = parser.parse_args()  # Parse the arguments
    if args.weight_grad:
        print("Weight gradient consideration is enabled.")
    else:
        print("Weight gradient consideration is disabled.")
    main(args)
