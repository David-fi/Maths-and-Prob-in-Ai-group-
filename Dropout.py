import numpy as np

class Dropout:
    """
    Author: Abdelrahmane Bekhli
    Date: 2024-18-11
    Description: This class performs dropouts.
    """
    def __init__(self, dropoutRate, mask=None, training=True, seed=None):
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
            
        self.dropoutRate = dropoutRate
        self.mask = mask
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
            self.mask = np.random.rand(*x.shape) > self.dropoutRate
            assert self.mask.shape == x.shape, f"Dropout mask shape {self.mask.shape} does not match input shape {x.shape}."
            return x * self.mask / (1 - self.dropoutRate)
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
        if self.mask is None:
            raise ValueError("Dropout mask is not initialized. Ensure dropoutForward is called during forward pass.")
        #assert dout.shape == self.mask.shape, f"Gradient shape {dout.shape} does not match dropout mask shape {self.mask.shape}."
        return dout * self.mask / (1 - self.dropoutRate)

    def setMode(self, mode):
        """
        Set the mode for the network: 'train' or 'test'
        Args:
            mode (str): Either 'train' or 'test'.
        """
        if mode == 'train':
            self.training = True
        elif mode == 'test':
            self.training = False
        else:
            raise ValueError("Mode can only be 'train' or 'test'")
        self.mask = None  # reset mask when changing modes
