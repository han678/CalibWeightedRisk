import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from common import get_loss_function_and_name, post_hoc_args
from loss.aurc import get_score_function, zero_one_loss
from loss.ece import ECELoss, Ece
from loss.ece_kde import get_bandwidth, get_ece_kde
from posthoc.metacal import MetaCalCoverageAcc, MetaCalMisCoverage
from posthoc.ts import TemperatureScale
from utils import Logger
from utils.aurc_estimators import get_em_AURC
from utils.metrics import get_brier_score


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
def evaluate_from_logits(train_loader, val_loader, device, score_func="MSP", temp_scale=False, temp_criterion=nn.CrossEntropyLoss()):
    all_logits, all_probs, all_targets, all_confidences = [], [], [], []
    total_correct_1, total_correct_5 = 0, 0
    score_func = get_score_function(score_func, input_is_softmax=False)
    eps = 1e-9

    # Apply temperature scaling if enabled
    if temp_scale:
        ts_model = TemperatureScale()
        ts_model.set_temp_from_loader(train_loader, temp_criterion)

    with torch.no_grad():
        for _, (logits, targets) in enumerate(val_loader):
            logits, targets = logits.to(device), targets.to(device)
            logits = logits if not temp_scale else ts_model.temp_scale(logits)
            probs = F.softmax(logits, dim=1) 
            probs = torch.clamp(probs, min=eps, max=1 - eps)
            confidence = score_func(logits)
            _, pred = probs.topk(5, 1, largest=True, sorted=True)
            correct = pred.eq(targets.view(targets.size(0), -1).expand_as(pred)).float()
            total_correct_1 += correct[:, :1].sum().item()
            total_correct_5 += correct[:, :5].sum().item()
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_targets.append(F.one_hot(targets, num_classes=probs.shape[1]).to(device).cpu().numpy())
            all_confidences.append(confidence.cpu().numpy())

    # Aggregate results
    all_logits = np.vstack(all_logits)
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    all_confidences = np.hstack(all_confidences)
    loss = zero_one_loss(all_probs, all_targets)
    top1_acc = 100. * total_correct_1 / len(val_loader.dataset)
    top5_acc = 100. * total_correct_5 / len(val_loader.dataset)
    # get_cwece = ClasswiseECELoss(n_bins=15)
    ece = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=False, classwise=False, is_input_softmax=False)
    ece_adaptive = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=True, classwise=False, is_input_softmax=False)
    cwece = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=False, classwise=True, is_input_softmax=False)
    cwece_adaptive = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=True, classwise=True, is_input_softmax=False)
    cwece_adaptive.set_bins(torch.from_numpy(all_confidences))
    ece_adaptive.set_bins(torch.from_numpy(all_confidences))
    # Compute additional metrics

    bandwidth = 0.02
    # bandwidth = get_bandwidth(all_probs, device='cpu')
    result = {
        "acc_1": top1_acc,
        "acc_5": top5_acc,
        "l1_ce": get_ece_kde(all_probs, all_targets, bandwidth=bandwidth, p=1, mc_type='canonical', device='cpu') * 100.0,
        "l2_ce": get_ece_kde(all_probs, all_targets, bandwidth=bandwidth, p=2, mc_type='canonical', device='cpu') * 100.0,
        "brier_score": get_brier_score(all_probs, all_targets).item() * 100.0,
        "ece": ece(logits=torch.from_numpy(all_logits), labels=torch.from_numpy(all_targets)).item() * 100.0,
        "ece_a":  ece_adaptive(logits=torch.from_numpy(all_logits), labels=torch.from_numpy(all_targets)).item() * 100.0,
        "cwece": cwece(logits=torch.from_numpy(all_logits), labels=torch.from_numpy(all_targets)) * 100.0,
        "cwece_a": cwece_adaptive(logits=torch.from_numpy(all_logits), labels=torch.from_numpy(all_targets)) * 100.0,
        "aurc": get_em_AURC(residuals=loss, confidence=all_confidences),
        "temp": round(ts_model.temp.item(), 4) if temp_scale else None
    }
    return result

# Evaluate MetaCal methods
def evaluate_meta_cal(X1, Y1, X2, Y2, score_func="MSP", method="miscoverage", acc=0.85, alpha=0.05):
    score_func = get_score_function(score_func, input_is_softmax=True)
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
    
    # Compute additional metrics
    confidence = score_func(torch.from_numpy(probs)).detach().cpu().numpy()
    bandwidth = 0.02
    # bandwidth = get_bandwidth(probs, device='cpu')
    Y2_onehot = np.eye(probs.shape[1])[Y2]
    loss = zero_one_loss(probs, Y2_onehot)
    ece2 = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=False, classwise=False, is_input_softmax=True)
    ece_adaptive = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=True, classwise=False, is_input_softmax=True)
    cwece = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=False, classwise=True, is_input_softmax=True)
    cwece_adaptive = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=True, classwise=True, is_input_softmax=True)
    cwece_adaptive.set_bins(torch.from_numpy(confidence))
    ece_adaptive.set_bins(torch.from_numpy(confidence))
    result = {
        "acc_1": top1_acc,
        "acc_5": top5_acc,
        "l1_ce": get_ece_kde(probs, Y2_onehot, bandwidth=bandwidth, p=1, mc_type='canonical', device='cpu') * 100.0,
        "l2_ce": get_ece_kde(probs, Y2_onehot, bandwidth=bandwidth, p=2, mc_type='canonical', device='cpu') * 100.0,
        "brier_score": get_brier_score(probs, Y2_onehot).item() * 100.0,
        "ece": ece2(logits=torch.from_numpy(probs), labels=torch.from_numpy(Y2_onehot)).item()* 100.0,
        "ece_a": ece_adaptive(logits=torch.from_numpy(probs), labels=torch.from_numpy(Y2_onehot)).item() * 100.0,
        "cwece": cwece(logits=torch.from_numpy(probs), labels=torch.from_numpy(Y2_onehot)) * 100.0,
        "cwece_a": cwece_adaptive(logits=torch.from_numpy(probs), labels=torch.from_numpy(Y2_onehot))* 100.0,
        "aurc": get_em_AURC(residuals=loss, confidence=confidence),
        "temp": None
    }
    return result


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('==> Preparing dataset %s' % args.dataset)
    root = f"train_output_ph/{args.dataset}/" 
    fig_root = f"finetune_rc/{args.dataset}/"
    fig_output_path = fig_root + "seed" + str(args.seed) if args.loss_type != "aurc" else fig_root + "/" + args.score_function + "seed" + str(args.seed)
    output_path = root + "/seed" + str(args.seed) if args.loss_type != "aurc" else root + args.score_function + "seed" + str(args.seed)
    if not os.path.isdir(output_path):
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(fig_output_path, exist_ok=True)

    save_npz_path = f"train_output/npz/{args.dataset}/{args.score_function}/" if args.loss_type == "aurc" else f"finetune/npz/{args.dataset}/"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Evaluate metrics
    score_function = args.score_function
    loss_type = args.loss_type
    regularization_strength = args.regularization_strength
    loss, _ = get_loss_function_and_name(args, loss_type, score_function, input_is_softmax=False, regularization_strength=regularization_strength)
    file_path = os.path.join(output_path, f'{args.arch}_{loss_type}.txt') if args.loss_type != "aurc" else os.path.join(output_path, f'{args.arch}_{loss_type}_rs{regularization_strength}.txt')
    # Load logits
    loss_name = args.loss_type if args.loss_type != "aurc" else args.loss_type + f"_{args.gamma}_rs{regularization_strength}"
    train_file = f'{save_npz_path}{args.arch}_{loss_name}_best_seed{args.seed}_train_logits.npz'
    test_file = f'{save_npz_path}{args.arch}_{loss_name}_best_seed{args.seed}_test_logits.npz'
    if not os.path.isfile(train_file) or not os.path.isfile(test_file):
        raise FileNotFoundError(f"Logits files not found: {train_file} or {test_file}")
    train_loader = load_logits_as_dataloader(train_file, batch_size=128, shuffle=False, device=device)
    test_loader = load_logits_as_dataloader(test_file, batch_size=128, shuffle=False, device=device)

    results = evaluate_from_logits(train_loader, test_loader, device, score_func="MSP", temp_scale=False, temp_criterion=loss)

    # Log results
    title = f"{args.dataset}-{loss_name}-{args.arch}"
    logger = Logger(file_path, title=title)
    add_names = ["Val Acc1", "Val Acc5"] if args.dataset in ['cifar100', 'imagenet'] else ["Val Acc"]
    logger.set_names(['Method'] + add_names + ["l1CE", "l2CE", 'Brier', "ece", "ece_a", "cwece", "cwece_a", 'AURC', "temp"])
    model_info = (
        list(results.values())
        if args.dataset in ['cifar100', 'imagenet']
        else list(results.values())[:1] + list(results.values())[2:]
    )
    logger.append(["None"] + model_info)

    # Temperature scaling
    ece_loss = ECELoss(n_bins=15, p=1)
    results_ls = evaluate_from_logits(train_loader, test_loader, device, temp_scale=True, score_func="MSP", temp_criterion=ece_loss)
    add_ls_res = list(results_ls.values()) if args.dataset in ['cifar100', 'imagenet'] else list(
        results_ls.values())[:1] + list(results_ls.values())[2:]
    logger.append(["ts"] + add_ls_res)

    # Meta-calibration
    X1, Y1 = load_logits_as_numpy(train_file)
    X2, Y2 = load_logits_as_numpy(test_file)
    acc = 0.97 if args.dataset == 'cifar10' else 0.87 if args.dataset == 'cifar100' else 0.85

    # Meta-cal miscoverage
    try:
        results_miscov = evaluate_meta_cal(X1, Y1, X2, Y2, score_func="MSP", method="miscoverage", alpha=0.05)
        add_miscov_res = list(results_miscov.values()) if args.dataset in ['cifar100', 'imagenet'] else list(
            results_miscov.values())[:1] + list(results_miscov.values())[2:]
        logger.append(["MetaCalMisCoverage"] + add_miscov_res)
    except Exception as e:
        print(f"An error occurred during meta-calibration miscoverage evaluation: {e}")

    # Meta-cal accuracy
    while acc <= 1.0:
        try:
            results_miscov = evaluate_meta_cal(X1, Y1, X2, Y2, score_func="MSP", method="coverageacc", acc=acc)
            add_miscov_res = list(results_miscov.values()) if args.dataset in ['cifar100', 'imagenet'] else list(
                results_miscov.values())[:1] + list(results_miscov.values())[2:]
            logger.append(["MetaCalCoverageAcc"] + add_miscov_res)
            break  # Exit the loop if successful
        except Exception as e:
            print(f"An error occurred during meta-calibration coverage accuracy evaluation: {e}")
            acc += 0.005  # Increment the accuracy threshold
            print(f"Increasing acc to {acc} and retrying...")

    logger.close()

if __name__ == '__main__':
    # Define arguments
    args = post_hoc_args()
    if args.dataset == 'tiny-imagenet':
        args.regularization_strength = 0.05
    elif args.dataset == 'cifar100':
        args.regularization_strength = 0.01
    elif args.dataset == 'cifar10':
        args.regularization_strength = 0.05
    else:
        pass
    if args.loss_type == "aurc":
        for args.score_function in ["MSP", "NegEntropy", "SoftmaxMargin"]:
            main(args)
    else:
        args.score_function = "MSP"
        main(args)
