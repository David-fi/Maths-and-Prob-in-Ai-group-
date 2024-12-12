#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Edited by Paula on Fri Dec 6 21:05:08 2024
"""

import numpy as np
class SoftmaxLayer:
    
    def __init__(self):
        # Prepare to store the output of the softmax function
        self.output = None

    @staticmethod
    def softmaxForward(logits):
        """
        Perform the forward pass to calculate softmax probabilities.
    
        Parameters:
            logits (np.array): Scores from the previous layer, shaped (batch_size, num_classes).
            
        Returns:
            np.array: Probabilities for each class, same shape as input.
        """
        # Subtract the max to keep numbers stable
        z_max = np.max(logits, axis=1, keepdims=True)
        shifted_logits = logits - z_max
        exp_shifted = np.exp(shifted_logits)
        
        # Divide by sum of exponents to get probabilities
        sum_exp = np.sum(exp_shifted, axis=1, keepdims=True)
        output = exp_shifted / sum_exp
        return output

    @staticmethod
    def softmaxBackward(output, true_labels):
        """
        Perform the backward pass to calculate gradient of the loss.
    
        Parameters:
            output (np.array): Probabilities from softmaxForward, shaped (batch_size, num_classes).
            true_labels (np.array): One-hot encoded true class labels, shaped (batch_size, num_classes).
            
        Returns:
            np.array: Gradient of the loss with respect to logits, same shape as input.
        """
        # Get the number of samples to average the gradient
        num_samples = true_labels.shape[0]
        
        # Calculate the gradient for softmax combined with cross-entropy
        gradient = (output - true_labels) / num_samples
        return gradient
