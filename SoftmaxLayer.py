import torch

class SoftmaxLayer:
    
    def __init__(self):
        # Prepare to store the output of the softmax function
        self.output = None

    @staticmethod
    def softmaxForward(logits):
        """
        Perform the forward pass to calculate softmax probabilities.
    
        Parameters:
            logits (torch.Tensor): Scores from the previous layer, shaped (batch_size, num_classes).
            
        Returns:
            torch.Tensor: Probabilities for each class, same shape as input.
        """
        # Subtract the max to keep numbers stable
        z_max = torch.max(logits, dim = 1, keepdim = True).values
        shifted_logits = logits - z_max
        exp_shifted = torch.exp(shifted_logits)
        
        # Divide by sum of exponents to get probabilities
        sum_exp = torch.sum(exp_shifted, dim = 1, keepdims = True)
        output = exp_shifted / sum_exp
        return output

    @staticmethod
    def softmaxBackward(output, true_labels):
        """
        Perform the backward pass to calculate gradient of the loss.
    
        Parameters:
            output (torch.Tensor): Probabilities from softmaxForward, shaped (batch_size, num_classes).
            true_labels (torch.Tensor): One-hot encoded true class labels, shaped (batch_size, num_classes).
            
        Returns:
            torch.Tensor: Gradient of the loss with respect to logits, same shape as input.
        """
        # Get the number of samples to average the gradient
        num_samples = true_labels.shape[0]
        
        # Calculate the gradient for softmax combined with cross-entropy
        gradient = (output - true_labels) / num_samples
        return gradient