from __future__ import print_function, absolute_import

import numpy as np

def get_brier_score(probs, targets):
    """
    Compute the Brier score for probabilistic predictions using NumPy.
    
    Args:
        probs (numpy.ndarray): The probabilities of each class. Shape (N, C) where C is number of classes.
        targets (numpy.ndarray): The one-hot encoded true labels. Shape (N, C) where C is number of classes.
        
    Returns:
        float: The Brier score for the predictions.
    """
    differences = probs - targets
    squared_differences = differences ** 2
    score = np.mean(squared_differences)
    return score


def get_ece_score(probs, targets, n_bins=15):
    """
    Calculate the top label Expected Calibration Error (ECE).
    
    Args:
        probs (np.ndarray): probs or predicted probabilities, shape (N, C), where C is number of classes.
        targets (np.ndarray): True labels or one-hot encoded labels, shape (N,) or (N, C).
        n_bins (int): Number of bins to use for ECE calculation.

    Returns:
        float: The ECE score.
    """
    # Convert to NumPy arrays
    probs = np.asarray(probs)
    targets = np.asarray(targets)
    
    # If targets are one-hot encoded, convert them to class indices
    if targets.ndim > 1:
        targets = np.argmax(targets, axis=1)
    
    # Get the predicted class indices and their probabilities
    predicted_classes = np.argmax(probs, axis=1)
    predicted_probs = np.array([probs[i, predicted_classes[i]] for i in range(probs.shape[0])])
    
    # Initialize bins
    accuracy_bins = np.zeros(n_bins)
    confidence_bins = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)
    
    # Calculate the binning step size
    for bin_index in range(n_bins):
        lower_bound, upper_bound = bin_index / n_bins, (bin_index + 1) / n_bins
        
        for i in range(probs.shape[0]):
            if lower_bound < predicted_probs[i] <= upper_bound:
                bin_counts[bin_index] += 1
                if predicted_classes[i] == targets[i]:
                    accuracy_bins[bin_index] += 1
                confidence_bins[bin_index] += predicted_probs[i]
        
        # Calculate mean accuracy and confidence for non-empty bins
        if bin_counts[bin_index] != 0:
            accuracy_bins[bin_index] /= bin_counts[bin_index]
            confidence_bins[bin_index] /= bin_counts[bin_index]
    
    # Compute the ECE score
    ece = np.sum(bin_counts * np.abs(accuracy_bins - confidence_bins)) / np.sum(bin_counts)
    
    return ece