import torch

class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x, training):
        """
        Forward pass for dropout.
        Args:
            x (torch.Tensor): The input to the dropout layer.
            training (bool): Is the model being trained.
        Returns:
            torch.Tensor: Output after applying dropout.
        """    
        if training:
            self.mask = (torch.rand_like(x) > self.dropout_rate).float()
            return x * self.mask / (1 - self.dropout_rate)
        return x

    def backward(self, dout):
        """
        Backward pass for dropout.
        Args:
            dout (torch.Tensor): The gradient from the next layer.
        Returns:
            torch.Tensor: Gradient after applying dropout mask.
        """
        return dout * self.mask / (1 - self.dropout_rate)