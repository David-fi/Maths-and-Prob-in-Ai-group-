import numpy as np
    
class Dropout:
    """
    Author: Abdelrahmane Bekhli
    Date: 2024-11-18
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
        x = np.array(x)
        
        if self.training == True:
            self.mask = np.random.rand(*x.shape) > self.dropoutRate
            return x * self.mask / (1 - self.dropoutRate)
        else:
            return x

    @staticmethod
    def dropoutBackward(self, dout):
        """
        Backward pass for dropout.
        Args:
            dout (numpy array): The gradient from the next layer.
        Returns:
            numpy array: Gradient after applying dropout mask.
        """
        return dout * self.mask / (1 - self.dropoutRate)
    
    
    @staticmethod
    def setMode(mode):
        """
        Set the mode for the network: 'train' or 'test'
        Args:
            mode (str): Either 'train' or 'test'.
        """
        if mode == 'train':
            Dropout.training = True
        elif mode == 'test':
            Dropout.training = False
        else:
            raise ValueError("Mode can only be 'train' or 'test'")
        Dropout.mask = None # reset mask when changing modes