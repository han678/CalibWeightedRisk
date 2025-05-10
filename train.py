from __future__ import print_function
import random
import numpy as np
import os
import torch.backends.cudnn as cudnn, torch.nn as nn

from common import common_args, get_loss_function_and_name
from models.utils import get_network
from utils import Logger
import torch
import torch.nn.functional as F
import timm
from loss.ece_kde import get_ece_kde
from utils.aurc_estimators import get_em_AURC
from utils.loaders import prepare_dataset
from loss.aurc import cross_entropy_loss, get_score_function
from utils.metrics import get_brier_score, get_ece_score
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


def evaluate(model, val_loader, score_func="NegEntropy", temp_scale=False, temp_criterion=nn.CrossEntropyLoss()):
    model.eval()
    all_probs, all_targets, all_confidences = [], [], []
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
            all_probs.append(probs.cpu().numpy())
            all_targets.append(F.one_hot(targets, num_classes=probs.shape[1]).to(device).cpu().numpy())
            all_confidences.append(confidence.cpu().numpy())
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    all_confidences = np.hstack(all_confidences)
    loss = cross_entropy_loss(all_probs, all_targets)
    top1_acc = 100. * total_correct_1 / len(val_loader.dataset)
    top5_acc = 100. * total_correct_5 / len(val_loader.dataset)
    brier_score = get_brier_score(all_probs, all_targets)
    result = {"acc_1": top1_acc, "acc_5": top5_acc}
    bandwidth = 0.02
    result['l1_ce'] = get_ece_kde(all_probs, all_targets, bandwidth=bandwidth, p=1, mc_type='canonical', device='cpu')
    result['l2_ce'] = get_ece_kde(all_probs, all_targets, bandwidth=bandwidth, p=2, mc_type='canonical', device='cpu')
    result["brier_score"] = brier_score
    result['ece'] = get_ece_score(all_probs, all_targets, n_bins=15)
    result["aurc"] = get_em_AURC(residuals=loss, confidence=all_confidences)
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
    test_loader = prepare_dataset(
        args.dataset, args.train_batch, False, args.workers, args.data_dir
    )
    root = "result/finetune/" if args.load_pretrain else "result/train/"
    save_model_path = "result/models/" + args.dataset
    save_npz_path = root + args.dataset
    output_path = root + args.dataset + "/seed" + str(args.seed) + args.score_function
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(save_model_path, exist_ok=True)
    if args.dataset != "tiny-imagenet":
        model = get_network(args)
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model.cuda())
            cudnn.benchmark = True
        if args.load_pretrain:
            print("==> Load pretrained model '%s'" % args.arch)
            model_path = args.model_dir + "/" + args.dataset + "/" + args.arch + f"_seed25.pth"
            model.load_state_dict(torch.load(model_path))
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
    results = evaluate(model, test_loader, score_func=score_function)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    if args.dataset == 'tiny-imagenet':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 60], gamma=0.1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

    title = f"{args.dataset}-{loss_type}-{args.arch}"
    # set up the output format
    loss_name = args.loss_type if args.loss_type != "aurc" else args.loss_type + f"_{args.gamma}_rs{regularization_strength}_{args.weight_grad}"
    logger = Logger(os.path.join(output_path, f'{args.arch}_{loss_name}.txt'), title=title)
    add_names = ["Train Acc1", "Train Acc5", "Val Acc1", "Val Acc5"] if args.dataset in ['cifar100', 'imagenet'] else ['Train Acc', "Val Acc"]
    logger.set_names(['Epoch', 'Train Loss'] + add_names + ["l1 CE", "l2 CE", 'Brier Score', "ece", 'AURC'])
    model_info = [0, None, None, None] + list(results.values()) if args.dataset in ['cifar100', 'imagenet'] else [0,None,None] + list(
        results.values())[:1] + list(results.values())[2:]
    # logger_post_hoc = Logger(os.path.join(output_path, f'{args.arch}_{loss_type}_ts.txt'), title=title)
    # logger_post_hoc.set_names(['Method', 'Train Loss'] + add_names + ["l1 CE", "l2 CE", 'Brier Score', "ece", 'AURC'])
    logger.append(model_info)
    # start training
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc_1, train_acc_5 = train(model, train_loader, loss, optimizer)
        print("lr: ", optimizer.param_groups[0]['lr'])
        lr_scheduler.step()
        results = evaluate(model, test_loader, score_func=score_function)
        add_res = [train_acc_5] + list(results.values()) if args.dataset in ['cifar100', 'imagenet'] else list(
            results.values())[:1] + list(results.values())[2:]  # remove the top 5 accuracy
        logger.append([int(epoch), train_loss, train_acc_1] + add_res)
        if epoch == args.epochs:
            export_logits(model, train_loader, f'{save_npz_path}/{args.arch}_{loss_name}_seed{args.seed}_train_logits', device)
            export_logits(model, test_loader, f'{save_npz_path}/{args.arch}_{loss_name}_seed{args.seed}_test_logits', device)
        # if epoch % 40 == 0: 
        #     export_logits(model, train_loader, f'{save_npz_path}/{args.arch}_{loss_name}_seed{args.seed}_train_logits_epoch{epoch}', device)
        #     export_logits(model, test_loader, f'{save_npz_path}/{args.arch}_{loss_name}_seed{args.seed}_test_logits_epoch{epoch}', device)
    # torch.save(model.state_dict(),  f"{save_model_path}/{args.arch}_{args.loss_type}_seed{args.seed}.pth")
    logger.close()
    # logger_post_hoc.close()


if __name__ == '__main__':
    parser = common_args()
    args = parser.parse_args()  # Parse the arguments
    if args.weight_grad:
        print("Weight gradient consideration is enabled.")
    else:
        print("Weight gradient consideration is disabled.")
    main(args)
