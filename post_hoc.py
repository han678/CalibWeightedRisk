import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from common import get_loss_function_and_name, get_model_names
from loss.aurc import get_score_function, zero_one_loss
from loss.ece import ECELoss
from loss.ece_kde import get_ece_kde
from posthoc.metacal import MetaCalCoverageAcc, MetaCalMisCoverage
from posthoc.ts import TemperatureScale
from utils import Logger
from utils.aurc_estimators import get_em_AURC
from utils.cwece import ClasswiseECELoss
from utils.metrics import get_brier_score, get_ece_score
from visualize.reliability_plots import reliability_plot

# Utility function to load logits as a DataLoader
def load_logits_as_dataloader(file_path, batch_size=64, shuffle=False, device='cpu'):
    data = np.load(file_path)
    logits = torch.from_numpy(data['logits']).to(device)
    targets = torch.from_numpy(data['targets']).to(device)
    dataset = TensorDataset(logits, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


# Utility function to load logits as NumPy arrays
def load_logits_as_numpy(file_path):
    data = np.load(file_path)
    logits = data['logits']
    targets = data['targets']
    return logits, targets


# Evaluate logits and compute metrics
def evaluate_from_logits(train_loader, val_loader, device, score_func="NegEntropy", temp_scale=False, temp_criterion=nn.CrossEntropyLoss()):
    all_probs, all_targets, all_confidences = [], [], []
    total_correct_1, total_correct_5 = 0, 0
    score_func = get_score_function(score_func)
    eps = 1e-9

    # Apply temperature scaling if enabled
    if temp_scale:
        ts_model = TemperatureScale()
        ts_model.set_temp_from_loader(train_loader, temp_criterion)

    with torch.no_grad():
        for _, (logits, targets) in enumerate(val_loader):
            logits, targets = logits.to(device), targets.to(device)
            probs = F.softmax(logits, dim=1) if not temp_scale else ts_model.predict_prob(logits)
            probs = torch.clamp(probs, min=eps, max=1 - eps)
            confidence = score_func(logits)
            _, pred = probs.topk(5, 1, largest=True, sorted=True)
            correct = pred.eq(targets.view(targets.size(0), -1).expand_as(pred)).float()
            total_correct_1 += correct[:, :1].sum().item()
            total_correct_5 += correct[:, :5].sum().item()
            all_probs.append(probs.cpu().numpy())
            all_targets.append(F.one_hot(targets, num_classes=probs.shape[1]).to(device).cpu().numpy())
            all_confidences.append(confidence.cpu().numpy())

    # Aggregate results
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    all_confidences = np.hstack(all_confidences)
    loss = zero_one_loss(all_probs, all_targets)
    top1_acc = 100. * total_correct_1 / len(val_loader.dataset)
    top5_acc = 100. * total_correct_5 / len(val_loader.dataset)
    get_cwece = ClasswiseECELoss(n_bins=15)
    # Compute additional metrics
    bandwidth = 0.02
    result = {
        "acc_1": top1_acc,
        "acc_5": top5_acc,
        "l1_ce": get_ece_kde(all_probs, all_targets, bandwidth=bandwidth, p=1, mc_type='canonical', device='cpu'),
        "l2_ce": get_ece_kde(all_probs, all_targets, bandwidth=bandwidth, p=2, mc_type='canonical', device='cpu'),
        "brier_score": get_brier_score(all_probs, all_targets),
        "ece": get_ece_score(all_probs, all_targets, n_bins=15),
        "cwece": get_cwece(torch.from_numpy(all_probs), torch.from_numpy(all_targets)),
        "aurc": get_em_AURC(residuals=loss, confidence=all_confidences),
        "temp": round(ts_model.temp.item(), 4) if temp_scale else None
    }
    return result


# Evaluate MetaCal methods
def evalute_meta_cal(X1, Y1, X2, Y2, score_func="NegEntropy", method="miscoverage", acc=0.85, alpha=0.05):
    score_func = get_score_function(score_func)
    eps = 1e-9

    # Choose MetaCal method
    if method == "miscoverage":
        model = MetaCalMisCoverage(alpha=alpha)
    else:
        model = MetaCalCoverageAcc(acc=acc)

    model.fit(X1, Y1)
    probs = model.predict(X2)
    probs = np.clip(probs, a_min=eps, a_max=1 - eps)
    pred = np.argsort(probs, axis=1)[:, ::-1][:, :5]  # Sort descending, take top 5
    correct = (pred == Y2[:, np.newaxis])

    # Compute top-1 and top-5 accuracy
    total_correct_1 = np.sum(correct[:, :1])
    total_correct_5 = np.sum(correct[:, :5])
    top1_acc = 100. * total_correct_1 / len(Y2)
    top5_acc = 100. * total_correct_5 / len(Y2)
    
    get_cwece = ClasswiseECELoss(n_bins=15)
    # Compute additional metrics
    confidence = score_func(torch.from_numpy(X2)).detach().cpu().numpy()
    bandwidth = 0.02
    Y2_onehot = np.eye(probs.shape[1])[Y2]
    loss = zero_one_loss(probs, Y2_onehot)
    result = {
        "acc_1": top1_acc,
        "acc_5": top5_acc,
        "l1_ce": get_ece_kde(probs, Y2_onehot, bandwidth=bandwidth, p=1, mc_type='canonical', device='cpu'),
        "l2_ce": get_ece_kde(probs, Y2_onehot, bandwidth=bandwidth, p=2, mc_type='canonical', device='cpu'),
        "brier_score": get_brier_score(probs, Y2_onehot),
        "ece": get_ece_score(probs, Y2_onehot, n_bins=15),
        "cwece": get_cwece(torch.from_numpy(probs), torch.from_numpy(Y2_onehot)),
        "aurc": get_em_AURC(residuals=loss, confidence=confidence),
        "temp": None
    }
    return result


# Parse command-line arguments
def post_hoc_args():
    model_names = get_model_names()
    parser = argparse.ArgumentParser(description='Post-hoc calibration')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', choices=model_names,
                        help='Model architecture: ' + ' | '.join(model_names) + ' (default: resnet50)')
    parser.add_argument('-d', '--dataset', default='cifar10', choices=['cifar10', 'svhn', 'cifar100', 'tiny-imagenet'])
    parser.add_argument('-j', '--workers', default=8, type=int)
    parser.add_argument('--loss_type', default='focal', type=str,
                        choices=['ce', 'aurc', "sele", "ece", "focal", "inverse_focal", "label_smoothing", "dual_focal", "ece_kde"])
    parser.add_argument('--score_function', default="NegEntropy", type=str)
    parser.add_argument('--seed', default=25, type=int)
    parser.add_argument('--gamma', default=0.5, type=float, help='Gamma parameter for AURC loss')
    parser.add_argument('--weight_grad', default=True, type=bool, help='Whether to consider weight gradient for AURC loss')
    parser.add_argument('--regularization_strength', default=0.05, type=float, help='Regularization strength for AURC loss')
    parser.add_argument('--train_batch', default=128, type=int, help='Batch size for training')
    parser.add_argument('--test_batch', default=128, type=int, help='Batch size for testing')
    return parser.parse_args()


def plot_reliability_curve(test_loader, device, save_fig=None):
    all_probs, all_targets, all_confidences = [], [], []

    with torch.no_grad():
        for _, (logits, targets) in enumerate(test_loader):
            logits, targets = logits.to(device), targets.to(device)
            probs = F.softmax(logits, dim=1)  # Compute softmax probabilities
            confidence = torch.max(probs, dim=1).values  # Use max softmax probability as confidence
            all_probs.append(probs.cpu().numpy())
            all_targets.append(F.one_hot(targets, num_classes=probs.shape[1]).to(device).cpu().numpy())
            all_confidences.append(confidence.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    all_confidences = np.hstack(all_confidences)

    true_labels = np.argmax(all_targets, axis=1)
    predicted_labels = np.argmax(all_probs, axis=1)

    reliability_plot(all_confidences, predicted_labels, true_labels, save_fig=save_fig)

def check_rc_plot(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('==> Preparing dataset %s' % args.dataset)
    root = "result/post_hoc/"
    output_path = root + args.dataset + "/seed" + str(args.seed)
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    save_npz_path = "result/train/" + args.dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Evaluate metrics
    score_function = args.score_function
    loss_type = args.loss_type
    regularization_strength = args.regularization_strength
    loss, loss_name = get_loss_function_and_name(args, loss_type, score_function, input_is_softmax=False, regularization_strength=regularization_strength)
    loss_name = args.loss_type if args.loss_type != "aurc" else args.loss_type + f"_{args.gamma}_rs{regularization_strength}_{args.weight_grad}"
    # Load logits
    train_file = f'{save_npz_path}/{args.arch}_{loss_name}_seed{args.seed}_train_logits.npz'
    test_file = f'{save_npz_path}/{args.arch}_{loss_name}_seed{args.seed}_test_logits.npz'
    test_loader = load_logits_as_dataloader(test_file, batch_size=128, shuffle=False, device=device)

    # Plot reliability curve
    fig_name = os.path.join(output_path, f'{args.arch}_{args.loss_type}_seed{args.seed}_rc.png')
    plot_reliability_curve(test_loader, device, save_fig=fig_name)

# Main function
def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('==> Preparing dataset %s' % args.dataset)
    root = "result/post_hoc/"
    output_path = root + args.dataset + "/seed" + str(args.seed)
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)

    save_npz_path = "result/train/" + args.dataset
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Evaluate metrics
    score_function = args.score_function
    loss_type = args.loss_type
    regularization_strength = args.regularization_strength
    loss, loss_name = get_loss_function_and_name(args, loss_type, score_function, input_is_softmax=False, regularization_strength=regularization_strength)
    loss_name = args.loss_type if args.loss_type != "aurc" else args.loss_type + f"_{args.gamma}_rs{regularization_strength}_{args.weight_grad}"
    # Load logits
    train_file = f'{save_npz_path}/{args.arch}_{loss_name}_seed{args.seed}_train_logits.npz'
    test_file = f'{save_npz_path}/{args.arch}_{loss_name}_seed{args.seed}_test_logits.npz'
    train_loader = load_logits_as_dataloader(train_file, batch_size=128, shuffle=False, device=device)
    test_loader = load_logits_as_dataloader(test_file, batch_size=128, shuffle=False, device=device)

    # Plot reliability curve
    fig_name = os.path.join(output_path, f'{args.arch}_{args.loss_type}_seed{args.seed}_rc.png')
    plot_reliability_curve(test_loader, device, save_fig=fig_name)

    results = evaluate_from_logits(train_loader, test_loader, device, score_func=score_function, temp_scale=False, temp_criterion=loss)

    # Log results
    title = f"{args.dataset}-{loss_name}-{args.arch}"
    logger = Logger(os.path.join(output_path, f'{args.arch}_{loss_type}.txt'), title=title)
    add_names = ["Val Acc1", "Val Acc5"] if args.dataset in ['cifar100', 'imagenet'] else ["Val Acc"]
    logger.set_names(['Method'] + add_names + ["l1 CE", "l2 CE", 'Brier Score', "ece", "cwece", 'AURC', "temp"])
    model_info = (
        list(results.values())
        if args.dataset in ['cifar100', 'imagenet']
        else list(results.values())[:1] + list(results.values())[2:]
    )
    logger.append(["None"] + model_info)

    # Temperature scaling
    ece_loss = ECELoss(n_bins=15, p=1)
    results_ls = evaluate_from_logits(train_loader, test_loader, device, temp_scale=True, score_func=score_function, temp_criterion=ece_loss)
    add_ls_res = list(results_ls.values()) if args.dataset in ['cifar100', 'imagenet'] else list(
        results_ls.values())[:1] + list(results_ls.values())[2:]
    logger.append(["ts"] + add_ls_res)

    # Meta-calibration
    X1, Y1 = load_logits_as_numpy(train_file)
    X2, Y2 = load_logits_as_numpy(test_file)
    acc = 0.97 if args.dataset == 'cifar10' else 0.87 if args.dataset == 'cifar100' else 0.85

    # # Meta-cal miscoverage
    try:
        results_miscov = evalute_meta_cal(X1, Y1, X2, Y2, score_func=score_function, method="miscoverage", alpha=0.05)
        add_miscov_res = list(results_miscov.values()) if args.dataset in ['cifar100', 'imagenet'] else list(
            results_miscov.values())[:1] + list(results_miscov.values())[2:]
        logger.append(["MetaCalMisCoverage"] + add_miscov_res)
    except Exception as e:
        print(f"An error occurred during meta-calibration miscoverage evaluation: {e}")

    # Meta-cal accuracy
    try:
        results_acc = evalute_meta_cal(X1, Y1, X2, Y2, score_func=score_function, method="coverageacc", acc=acc)
        add_acc_res = list(results_acc.values()) if args.dataset in ['cifar100', 'imagenet'] else list(
            results_acc.values())[:1] + list(results_acc.values())[2:]
        logger.append(["MetaCalCoverageAcc"] + add_acc_res)
    except Exception as e:
        print(f"An error occurred during meta-calibration accuracy evaluation: {e}")

    logger.close()


if __name__ == '__main__':
    # Define arguments
    args = post_hoc_args()

    # Seeds, architectures, loss types, and datasets
    seeds = [40, 41, 42, 25, 52] # , 41, 42, 25, 52
    archs = ['resnet50', 'wrn', 'resnet110', 'PreResNet56', 'densenet121']
    
    datasets = ['cifar10', 'cifar100']
    args.gamma = 0.5
    # Iterate over all combinations
    for args.seed in seeds:
        for args.arch in archs:
            for args.dataset in datasets:
                args.regularization_strength = 0.05 if args.dataset == "cifar10" else 0.01
                loss_types = [f'aurc_0.5_True', 'ce', 'focal', 'inverse_focal', 'dual_focal', 'ece_kde'] if args.dataset == "cifar10" else [f'aurc_0.5_rs{args.regularization_strength}_True', 'ce', 'focal', 'inverse_focal', 'dual_focal', 'ece_kde']#['aurc', 'ce', 'focal', 'inverse_focal', 'dual_focal', 'ece_kde']
                for args.loss_type in loss_types:
                    check_rc_plot(args)
                    # main(args)