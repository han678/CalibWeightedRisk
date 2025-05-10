'''
This file contains methods for generating calibration-related plots, e.g., reliability plots and confidence histograms.

References:
[1] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks.
    arXiv preprint arXiv:1706.04599, 2017.
'''

import math
from matplotlib.cm import get_cmap
import numpy as np
import matplotlib.pyplot as plt
import torch
plt.rcParams.update({'font.size': 20})

# Keys used for bin dictionaries
COUNT = 'count'
CONF = 'conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'


def _bin_initializer(bin_dict, num_bins=15):
    """Initialize bins for calibration metrics."""
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0


def _populate_bins(confs, preds, labels, num_bins=15):
    """Populate bins with confidence, accuracy, and count data."""
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil((num_bins * confidence) - 1))
        bin_dict[binn][COUNT] += 1
        bin_dict[binn][CONF] += confidence
        bin_dict[binn][ACC] += (1 if label == prediction else 0)

    for binn in range(num_bins):
        if bin_dict[binn][COUNT] == 0:
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / float(bin_dict[binn][COUNT])
    return bin_dict


def compute_calibration_metrics(bin_dict, num_bins):
    """Compute calibration metrics such as ECE and MCE."""
    total_samples = sum(bin_dict[b][COUNT] for b in range(num_bins))
    ece, mce = 0.0, 0.0

    for b in range(num_bins):
        if bin_dict[b][COUNT] > 0:
            gap = abs(bin_dict[b][BIN_ACC] - bin_dict[b][BIN_CONF])
            ece += (gap * bin_dict[b][COUNT]) / total_samples
            mce = max(mce, gap)

    return ece, mce


def reliability_plot(confs, preds, labels, num_bins=15, save_fig=".figs/reliability_plot.png"):
    '''
    Method to draw a reliability plot from a model's predictions and confidences.
    '''
    # Populate bins and compute calibration metrics
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    ece, mce = compute_calibration_metrics(bin_dict, num_bins)

    # Extract bin data
    accuracies = [bin_dict[i][BIN_ACC] for i in range(num_bins)]
    confidences = [bin_dict[i][BIN_CONF] for i in range(num_bins)]
    bins = [(i / float(num_bins)) for i in range(num_bins + 1)]

    # Prepare for plotting
    bin_size = 1.0 / num_bins
    positions = np.array(bins[:-1]) + bin_size / 2.0
    gap_color = "orangered"   #  # Default to overconfidence color
    acc_color = "royalblue"  #"#E1EEBC"  # Fixed color code

    gap= [np.abs(accuracies[i] - confidences[i]) for i in range(num_bins)]
    # Plot reliability diagram
    plt.figure(figsize=(10, 8))
    plt.bar(positions, accuracies, bottom=0, width=bin_size, color=acc_color, edgecolor="white", label="Accuracy", alpha=1.0)
    plt.bar(positions, gap, bottom=np.minimum(accuracies, confidences), width=bin_size, color=gap_color, 
            hatch="/", edgecolor="white", alpha=0.5, label ="Gap")
    # plt.bar(positions, overconfidence_gap, bottom=accuracies, width=bin_size, color=gap_color_under, edgecolor="white", label="Overconfidence Gap")

    # Add y=x reference line
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")

    # Add labels, legend, and title
    plt.ylabel('Accuracy')
    plt.xlabel('Confidence')
    plt.legend(fontsize=20, loc='upper left')

    # Add ECE and MCE as text annotations
    plt.text(
        0.98, 0.02,
        f"ECE: {ece:.3f}\nMCE: {mce:.3f}",
        fontsize=22,
        transform=plt.gca().transAxes,
        color="black",
        ha="right",
        va="bottom",
        bbox=dict(facecolor='lightgray', edgecolor='black', boxstyle='round,pad=0.3')
    )

    # Save and show the plot
    plt.tight_layout()
    plt.savefig(save_fig, bbox_inches='tight')
    plt.show()

def confidence_histogram(confs, preds, labels, num_bins=15, save_fig=".figs/confidence_histogram.png"):
    '''
    Method to draw a confidence histogram showing the number of samples in each bin.
    '''
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    bns = [(i / float(num_bins)) for i in range(num_bins)]
    num_samples = len(labels)
    y = [(bin_dict[i][COUNT] / float(num_samples)) * 100 for i in range(num_bins)]

    plt.figure(figsize=(10, 8))
    plt.bar(bns, y, align='edge', width=0.05, color='blue', alpha=0.5, label='Percentage samples')
    plt.ylabel('Percentage of samples')
    plt.xlabel('Confidence')
    plt.title("Confidence Histogram")
    plt.tight_layout()
    plt.savefig(save_fig, bbox_inches='tight')
    plt.show()


def combined_reliability_diagram(confs, preds, labels, num_bins=15, save_fig=".figs/combined_reliability_diagram.png"):
    '''
    Method to draw a combined reliability diagram and confidence histogram.
    '''
    bin_dict = _populate_bins(confs, preds, labels, num_bins)
    ece, mce = compute_calibration_metrics(bin_dict, num_bins)

    bns = [(i / float(num_bins)) for i in range(num_bins)]
    acc_y = [bin_dict[i][BIN_ACC] for i in range(num_bins)]
    hist_y = [(bin_dict[i][COUNT] / float(len(labels))) * 100 for i in range(num_bins)]

    fig, ax = plt.subplots(2, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [3, 1]})

    # Reliability plot
    ax[0].bar(bns, bns, align='edge', width=0.05, color='pink', label='Expected')
    ax[0].bar(bns, acc_y, align='edge', width=0.05, color='blue', alpha=0.5, label='Actual')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Confidence')
    ax[0].legend()
    ax[0].set_title(f"Reliability Plot\nECE: {ece:.4f}, MCE: {mce:.4f}")

    # Confidence histogram
    ax[1].bar(bns, hist_y, align='edge', width=0.05, color='blue', alpha=0.5, label='Percentage samples')
    ax[1].set_ylabel('Percentage of samples')
    ax[1].set_xlabel('Confidence')
    ax[1].set_title("Confidence Histogram")

    plt.tight_layout()
    plt.savefig(save_fig, bbox_inches='tight')
    plt.show()


def get_scores_and_labels(preds_dict, logits_key, labels_key):
    EPS = 1e-7

    logits = preds_dict[logits_key]
    labels = preds_dict[labels_key]

    scores = torch.softmax(logits, dim=1)
    scores = torch.clamp(scores, min=EPS, max=1 - EPS)

    return scores, labels


if __name__ == "__main__":
    # Example usage
    preds_dict = {
        'logits': torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.6, 0.4]]),
        'labels': torch.tensor([1, 0, 1])
    }
    logits_key = 'logits'
    labels_key = 'labels'

    scores, labels = get_scores_and_labels(preds_dict, logits_key, labels_key)
    scores = scores.numpy()
    labels = labels.numpy()

    reliability_plot(scores[:, 1], np.argmax(scores, axis=1), labels)
    confidence_histogram(scores[:, 1], np.argmax(scores, axis=1), labels)
    combined_reliability_diagram(scores[:, 1], np.argmax(scores, axis=1), labels)