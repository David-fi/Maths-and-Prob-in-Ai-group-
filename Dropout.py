import numpy as np

class Dropout:
    """
    Author: Abdelrahmane Bekhli
    Date: 2024-18-11
    Description: This class performs dropouts.
    """
    def __init__(self, dropoutRate, training=True, seed=None):
        """ 
        Initialize the Dropout layer. 
        Args: 
            dropoutRate (float): The probability of dropping out a unit. 
            seed (int, optional): Random seed for reproducibility. 
        """
        if not (0 <= dropoutRate < 1):
            raise ValueError("Dropout rate must be between 0 and 1")
        if seed is not None:
            np.random.seed(seed)

        self.maskCache = []            
        self.dropoutRate = dropoutRate
        self.training = training

    def dropoutForward(self, x):
        """
        Forward pass for dropout.
        Args:
            x (numpy array): The input to the dropout layer.
        Returns:
            numpy array: Output after applying dropout.
        """
        if not isinstance(x, np.ndarray):
            raise TypeError(f"Input must be a numpy array, but got {type(x).__name__}.")
        
        if self.training:
            mask = np.random.rand(*x.shape) > self.dropoutRate
            self.maskCache.append(mask)
            assert mask.shape == x.shape, f"Dropout mask shape {mask.shape} does not match input shape {x.shape}."
            return x * mask / (1 - self.dropoutRate)
        else:
            return x

    def dropoutBackward(self, dout):
        """
        Backward pass for dropout.
        Args:
            dout (numpy array): The gradient from the next layer.
        Returns:
            numpy array: Gradient after applying dropout mask.
        """
        if len(self.maskCache) == 0:
            raise ValueError("Dropout mask is not initialized. Ensure dropoutForward is called during forward pass.")

        return dout * self.maskCache.pop() / (1 - self.dropoutRate)

    def setMode(self, mode):
        """
        Set the mode for the network: 'train' or 'test'
        Args:
            mode (bool): Either 'train' or 'test'.
        """
        self.training = mode
        self.maskCache.clear()  # reset mask when changing modes
