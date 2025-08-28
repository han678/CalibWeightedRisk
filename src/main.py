from __future__ import print_function
import os
import random
import numpy as np
import torch
import torch.nn as nn
import time

import torch.nn.functional as F
from models.utils import get_network
from utils.loaders import prepare_dataset, train_val_split  
from utils import Logger
from loss.ece_kde import get_ece_kde
from loss.ece import Ece
from loss.aurc import get_score_function, cross_entropy_loss
from utils.aurc_estimators import get_em_AURC
from utils.metrics import get_brier_score
from config import (
    configure_training_params,
    get_save_criteria,
    get_loss_function,
    create_parser
)

def evaluate_simple(model, loader, criterion=None, return_ece=True):
    model.eval()
    device = next(model.parameters()).device
    correct_1, correct_5, val_loss, total = 0, 0, 0, 0
    all_logits, all_targets = [], []
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if targets.dim() > 1:
                targets = targets.squeeze()
            if targets.dtype != torch.long:
                targets = targets.long()
            logits = model(images)
            if criterion:
                val_loss += criterion(logits, targets).item()
            all_logits.append(logits)
            all_targets.append(targets)
            _, pred = logits.topk(5, 1)
            correct = pred.eq(targets.view(-1, 1).expand_as(pred)).float()
            correct_1 += correct[:, :1].sum().item()
            correct_5 += correct[:, :5].sum().item()
            total += targets.size(0)
    top1 = 100. * correct_1 / total
    top5 = 100. * correct_5 / total
    avg_loss = val_loss / len(loader) if criterion else None
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    if return_ece:
        ece_calculator = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=False, classwise=False, prob=False, return_per_class=False)
        ece_value = ece_calculator(logits=all_logits, labels=all_targets) * 100.0
        return {"loss": avg_loss, "acc_1": top1, "acc_5": top5, "ece": ece_value}
    else:
        return {"loss": avg_loss, "acc_1": top1, "acc_5": top5}

def evaluate(model, loader, criterion=None, score_func="MSP", temp_scale=False, temp_criterion=nn.CrossEntropyLoss()):
    """Full evaluation with all calibration metrics (use only at the end)."""
    model.eval()
    device = next(model.parameters()).device
    score_fn = get_score_function(score_func)
    eps = 1e-9
    if temp_scale:
        ts_model = ModelWithTemperature(model)
        ts_model.set_temp(loader, temp_criterion)
    logits_list, probs_list, targets_list, confidences = [], [], [], []
    correct_1, correct_5, val_loss = 0, 0, 0
    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            if targets.dim() > 1:
                targets = targets.squeeze()
            if targets.dtype != torch.long:
                targets = targets.long()
                
            logits = model(images)
            logits = logits if not temp_scale else ts_model.temp_scale(logits)
            if criterion is not None:
                val_loss += criterion(logits, targets).item()
            probs = torch.clamp(F.softmax(logits, dim=1), min=eps, max=1 - eps)
            confidence = score_fn(logits)
            _, pred = probs.topk(5, 1)
            correct = pred.eq(targets.view(-1, 1).expand_as(pred)).float()
            correct_1 += correct[:, :1].sum().item()
            correct_5 += correct[:, :5].sum().item()
            logits_list.append(logits.cpu().numpy())
            probs_list.append(probs.cpu().numpy())
            targets_list.append(F.one_hot(targets, num_classes=probs.shape[1]).cpu().numpy())
            confidences.append(confidence.cpu().numpy())
    logits_np = np.vstack(logits_list)
    probs_np = np.vstack(probs_list)
    targets_np = np.vstack(targets_list)
    conf_np = np.hstack(confidences)
    loss = cross_entropy_loss(probs_np, targets_np)
    top1 = 100. * correct_1 / len(loader.dataset)
    top5 = 100. * correct_5 / len(loader.dataset)
    avg_loss = val_loss / len(loader) if criterion else None
    ece = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=False, classwise=False, prob=False, return_per_class=False)
    ece_adaptive = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=True, classwise=False, prob=False, return_per_class=False)
    cwece = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=False, classwise=True, prob=False, return_per_class=False)
    cwece_adaptive = Ece(p=1, n_bins=15, version="not-our", adaptive_bins=True, classwise=True, prob=False, return_per_class=False)
    return {
        "loss": avg_loss, "acc_1": top1, "acc_5": top5,
        "l1_ce": get_ece_kde(probs_np, targets_np, bandwidth=0.02, p=1, mc_type='canonical', device='cpu') * 100.0,
        "l2_ce": get_ece_kde(probs_np, targets_np, bandwidth=0.02, p=2, mc_type='canonical', device='cpu') * 100.0,
        "brier_score": get_brier_score(probs_np, targets_np) * 100.0,
        "ece": ece(logits=torch.from_numpy(logits_np), labels=torch.from_numpy(targets_np)) * 100.0,
        "ece_a": ece_adaptive(logits=torch.from_numpy(logits_np), labels=torch.from_numpy(targets_np)) * 100.0,
        "cwece": cwece(logits=torch.from_numpy(logits_np), labels=torch.from_numpy(targets_np)) * 100.0,
        "cwece_a": cwece_adaptive(logits=torch.from_numpy(logits_np), labels=torch.from_numpy(targets_np)) * 100.0,
        "aurc": get_em_AURC(residuals=loss, confidence=conf_np),
    }

def train(model, train_loader, criterion, optimizer, scaler, use_amp=False):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        
        # Ensure targets are 1D for classification
        if targets.dim() > 1:
            targets = targets.squeeze()
        if targets.dtype != torch.long:
            targets = targets.long()
        
        optimizer.zero_grad()

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(f'Train Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.3f} | Acc: {100.*correct/total:.2f}%')

    accuracy = 100. * correct / total
    avg_loss = train_loss / len(train_loader)
    epoch_time = time.time() - start_time
    
    print(f'Train: Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% | Time: {epoch_time:.1f}s')
    
    return avg_loss, accuracy

def export_logits(model, loader, save_path, device):
    model.eval()
    logits, targets = [], []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            logits.append(model(data).cpu().numpy())
            targets.append(target.numpy())
    np.savez(save_path, logits=np.vstack(logits), targets=np.hstack(targets))

def build_paths(args, root="train_output"):
    if args.loss_type == "aurc":
        file_name = f"{args.arch}_{args.loss_type}_{args.score_function}"
    else:
        file_name = f"{args.arch}_{args.loss_type}"
    model_folder = os.path.join(root, args.dataset, "models", f"seed{args.seed}")
    npz_folder = os.path.join(root, args.dataset, "npz", f"seed{args.seed}")
    log_folder = os.path.join(root, args.dataset, "logs", f"seed{args.seed}")
    for path in [model_folder, npz_folder, log_folder]:
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(model_folder, f"{file_name}.pth")
        npz_path = os.path.join(npz_folder, f"{file_name}")
        log_path = os.path.join(log_folder, f"{file_name}.txt")
    return model_path, npz_path, log_path

def setup_optimizer_scheduler(model, args):
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    if args.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=args.milestones, 
            gamma=args.lr_gamma
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    return optimizer, scheduler

def main(args):
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"==> Preparing dataset {args.dataset}")
    train_loader, val_loader = train_val_split(args)
    num_workers = args.workers
    test_loader = prepare_dataset(
        args.dataset, 
        batch_size=args.batch_size, 
        load_train=False, 
        num_workers=num_workers, 
        data_dir=args.data_dir, 
        return_loader=True, 
        arch=args.arch
    )

    model_path, npz_path, log_path = build_paths(args)
    model = get_network(args).to(device)
    score_func = args.score_function
    loss_fn = get_loss_function(args, args.loss_type, score_func, is_prob=False)
    optimizer, scheduler = setup_optimizer_scheduler(model, args)
    scaler = torch.cuda.amp.GradScaler()
    train_log = Logger(log_path, title=f"{args.dataset}-{args.loss_type}-{args.arch}")
    train_log.set_names([
        'Epoch', "Train Acc1", 'Train Loss', "Val Loss", "Val Acc1", "Val Acc5", "Val ECE",
        "Test Acc1", "Test Acc5", "Test ECE", "Test ECE_A", "Test CWECE", "Test CWECE_A",
        "Test L1_CE", "Test L2_CE", "Test Brier", "Test AURC"
    ])
    best_val_loss = float('inf')
    save_criteria = get_save_criteria(args.dataset, args.epochs)

    print(f"Starting training for {args.epochs} epochs...")
    print(f"Architecture: {args.arch}, Dataset: {args.dataset}, Loss: {args.loss_type}")
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        print(f'\nEpoch {epoch}/{args.epochs}:')
        # Training phase
        train_start_time = time.time()
        train_loss, train_acc1 = train(model, train_loader, loss_fn, optimizer, scaler, use_amp=args.use_amp)
        train_time = time.time() - train_start_time

        scheduler.step()
        # Validation phase
        eval_start_time = time.time()
        val_res = evaluate_simple(model, val_loader, loss_fn)
        test_res = evaluate(model, test_loader, loss_fn, score_func=args.score_function)
        train_log.append([
            epoch, train_acc1, train_loss, val_res["loss"], val_res["acc_1"], val_res["acc_5"], val_res["ece"],
            test_res["acc_1"], test_res["acc_5"], test_res["ece"], test_res["ece_a"], test_res["cwece"], test_res["cwece_a"],
            test_res["l1_ce"], test_res["l2_ce"], test_res["brier_score"], test_res["aurc"]
        ])
        should_save = epoch >= save_criteria['warmup_epoch']
        if val_res["loss"] < best_val_loss:
            best_val_loss = val_res["loss"]
            if should_save:
                torch.save(model.state_dict(), model_path)
                print(f"*** Best model saved! Val loss: {best_val_loss:.4f} ***")
        eval_time = time.time() - eval_start_time
        # Total epoch time
        epoch_time = time.time() - epoch_start_time

        print(f"Epoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc1:.2f}%")
        print(f"  Val Loss: {val_res['loss']:.4f} | Val Acc@1: {val_res['acc_1']:.2f}% | Val Acc@5: {val_res['acc_5']:.2f}%")
        print(f"  Val ECE: {val_res['ece']:.2f}%")
        print(f"  Time - Train: {train_time:.1f}s | Eval: {eval_time:.1f}s | Total: {epoch_time:.1f}s")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Estimate remaining time
        if epoch > 1:
            avg_epoch_time = epoch_time
            remaining_epochs = args.epochs - epoch
            estimated_remaining = remaining_epochs * avg_epoch_time
            hours, remainder = divmod(estimated_remaining, 3600)
            minutes, seconds = divmod(remainder, 60)
            print(f"  Estimated remaining: {int(hours)}h {int(minutes)}m {int(seconds)}s")
    train_log.close()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    export_logits(model, train_loader, f'{npz_path}_train_logits', device)
    export_logits(model, test_loader, f'{npz_path}_test_logits', device)

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    args = configure_training_params(args)
    if args.arch in ['vit_small', 'vit_base', 'swin_tiny', 'swin_base']:
        args.load_pretrained = 1  
    main(args)
