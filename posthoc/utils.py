import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from loss.aurc import get_score_function, cross_entropy_loss
from loss.ece import Ece
from loss.ece_kde import get_ece_kde
from utils.metrics import get_brier_score
from posthoc.ts import TemperatureScale
import os 

# Regularization strength per dataset
REG_STRENGTH = {
    'tiny-imagenet': 0.05,
    'cifar100': 0.01,
    'cifar10': 0.05,
}

# Score functions for each loss type
LOSS_SCORE_FUNCTIONS = {
    'aurc': ["MSP", "NegEntropy", "SoftmaxMargin"],
    # Other loss types can be added if needed
}

# Default score function for non-aurc
DEFAULT_SCORE_FUNCTION = "MSP"

# MetaCal default acc per dataset
METACAL_ACC = {
    'cifar10': 0.97,
    'cifar100': 0.87,
    'tiny-imagenet': 0.85,
}


def export_logits(model, loader, save_path, device):
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            logits.append(model(data).cpu().numpy())
            targets.append(target.numpy())
    np.savez(save_path, logits=np.vstack(logits), targets=np.hstack(targets))


def load_logits_as_dataloader(logits_file, batch_size=128, shuffle=False, device='cpu'):
    data = np.load(logits_file)
    logits = torch.from_numpy(data['logits']).to(device)
    targets = torch.from_numpy(data['targets']).to(device)
    dataset = TensorDataset(logits, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def load_logits_as_numpy(logits_file):
    data = np.load(logits_file)
    return data['logits'], data['targets']

def evaluate_logits_metrics(loader, device, score_function="MSP", temp_scale=False, temp_criterion=None, temp_train_loader=None):
    all_logits, all_probs, all_targets, all_confidences = [], [], [], []
    total_correct_1, total_correct_5 = 0, 0
    score_func = get_score_function(score_function, is_prob=False)
    eps = 1e-9

    if temp_scale:
        assert temp_train_loader is not None, "temp_train_loader must be provided for temperature scaling"
        ts_model = TemperatureScale()
        ts_model.set_temp_from_loader(temp_train_loader, temp_criterion)
    else:
        ts_model = None

    with torch.no_grad():
        for logits, targets in loader:
            logits, targets = logits.to(device), targets.to(device)
            if temp_scale:
                logits = ts_model.temp_scale(logits)
            probs = F.softmax(logits, dim=1)
            probs = torch.clamp(probs, min=eps, max=1 - eps)
            confidence = score_func(logits)
            _, pred = probs.topk(5, 1, largest=True, sorted=True)
            correct = pred.eq(targets.view(targets.size(0), -1).expand_as(pred)).float()
            total_correct_1 += correct[:, :1].sum().item()
            total_correct_5 += correct[:, :5].sum().item()
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_targets.append(F.one_hot(targets, num_classes=probs.shape[1]).cpu().numpy())
            all_confidences.append(confidence.cpu().numpy())

    all_logits = np.vstack(all_logits)
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    all_confidences = np.hstack(all_confidences)
    loss = cross_entropy_loss(all_probs, all_targets)
    top1 = 100. * total_correct_1 / len(loader.dataset)
    top5 = 100. * total_correct_5 / len(loader.dataset)
    bandwidth = 0.02
    ece = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=False, classwise=False, prob=False, return_per_class=False)
    ece_adaptive = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=True, classwise=False, prob=False, return_per_class=False)
    cwece = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=False, classwise=True, prob=False, return_per_class=False)
    cwece_adaptive = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=True, classwise=True, prob=False, return_per_class=False)
    result = {
        "acc_1": top1,
        "acc_5": top5,
        "l1_ce": get_ece_kde(all_probs, all_targets, bandwidth=bandwidth, p=1, mc_type='canonical', device='cpu') * 100.0,
        "l2_ce": get_ece_kde(all_probs, all_targets, bandwidth=bandwidth, p=2, mc_type='canonical', device='cpu') * 100.0,
        "brier_score": get_brier_score(all_probs, all_targets).item() * 100.0,
        "ece": ece(logits=torch.from_numpy(all_logits), labels=torch.from_numpy(all_targets)) * 100.0,
        "ece_a":  ece_adaptive(logits=torch.from_numpy(all_logits), labels=torch.from_numpy(all_targets)) * 100.0,
        "cwece": cwece(logits=torch.from_numpy(all_logits), labels=torch.from_numpy(all_targets)) * 100.0,
        "cwece_a": cwece_adaptive(logits=torch.from_numpy(all_logits), labels=torch.from_numpy(all_targets)) * 100.0,
        "temp": round(ts_model.temp.item(), 3) if temp_scale else None
    }
    return result

def evaluate_metacal_metrics(X_train, y_train, X_test, y_test, score_function="MSP", method="miscoverage", acc=0.85, alpha=0.05):
    score_func = get_score_function(score_function, is_prob=True)
    eps = 1e-9
    if method == "miscoverage":
        from posthoc.metacal import MetaCalMisCoverage
        model = MetaCalMisCoverage(alpha=alpha)
    else:
        from posthoc.metacal import MetaCalCoverageAcc
        model = MetaCalCoverageAcc(acc=acc)
    model.fit(X_train, y_train)
    probs = model.predict(X_test).astype(np.float32)
    probs = np.clip(probs, a_min=eps, a_max=1 - eps)
    pred = np.argsort(probs, axis=1)[:, ::-1][:, :5]
    correct = (pred == y_test[:, np.newaxis])
    total_correct_1 = np.sum(correct[:, :1])
    total_correct_5 = np.sum(correct[:, :5])
    top1 = 100. * total_correct_1 / len(y_test)
    top5 = 100. * total_correct_5 / len(y_test)
    bandwidth = 0.02
    y_test_onehot = np.eye(probs.shape[1])[y_test]
    ece = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=False, classwise=False, prob=True, return_per_class=False)
    ece_adaptive = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=True, classwise=False, prob=True, return_per_class=False)
    cwece = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=False, classwise=True, prob=True, return_per_class=False)
    cwece_adaptive = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=True, classwise=True, prob=True, return_per_class=False)
    result = {
        "acc_1": top1,
        "acc_5": top5,
        "l1_ce": get_ece_kde(probs, y_test_onehot, bandwidth=bandwidth, p=1, mc_type='canonical', device='cpu') * 100.0,
        "l2_ce": get_ece_kde(probs, y_test_onehot, bandwidth=bandwidth, p=2, mc_type='canonical', device='cpu') * 100.0,
        "brier_score": get_brier_score(probs, y_test_onehot).item() * 100.0,
        "ece": ece(logits=torch.from_numpy(probs), labels=torch.from_numpy(y_test_onehot)) * 100.0,
        "ece_a": ece_adaptive(logits=torch.from_numpy(probs), labels=torch.from_numpy(y_test_onehot)) * 100.0,
        "cwece": cwece(logits=torch.from_numpy(probs), labels=torch.from_numpy(y_test_onehot)) * 100.0,
        "cwece_a": cwece_adaptive(logits=torch.from_numpy(probs), labels=torch.from_numpy(y_test_onehot)) * 100.0,
        "temp": None
    }
    return result

