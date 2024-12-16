import numpy as np

class Dropout:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, x, training):
        """
        Forward pass for dropout.
        Args:
            x (numpy array): The input to the dropout layer.
            training (bool): Is the model being trained.
        Returns:
            numpy array: Output after applying dropout.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Input must be a numpy array, but got {type(x).__name__}.")
    
        if training:
            self.mask = np.random.rand(*x.shape) > self.dropout_rate
            return x * self.mask / (1 - self.dropout_rate)
        return x

    def backward(self, dout):
        """
        Backward pass for dropout.
        Args:
            dout (numpy array): The gradient from the next layer.
        Returns:
            numpy array: Gradient after applying dropout mask.
        """
        return dout * self.mask / (1 - self.dropout_rate)