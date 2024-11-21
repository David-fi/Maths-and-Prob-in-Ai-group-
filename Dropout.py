import numpy as np

class Dropout:
    """
    Author: Abdelrahmane Bekhli
    Date: 2024-11-18
    Description: This class performs dropouts.
    """
    def __init__(self, dropoutRate):
        if not (0 <= dropoutRate < 1):
            raise ValueError("Dropout rate must be between 0 and 1")
        self.dropoutRate = dropoutRate
        self.mask = None
        self.training = True
    
    def forward(self, x):
        """
        Forward pass for dropout.
        Args:
            input_data: The input to the dropout layer (numpy array).
        Returns:
            Output after applying dropout.
        """
        if self.training:
            self.mask = np.random.rand(*x.shape) > self.dropoutRate
            return x * self.mask / (1 - self.dropoutRate)
        else:
            return x
    
    def backward(self, dout):
        """
        Backward pass for dropout.
        Args:
            dout: The gradient from the next layer.
        Returns:
            Gradient after applying dropout mask.
        """
        return dout * self.mask / (1 - self.dropoutRate)
    
    def setMode(self, mode):
        """
        Set the mode for the network: 'train' or 'test'
        Args:
            mode: Either 'train' or 'test'
        """
        if mode == 'train':
            self.training = True
        elif mode == 'test':
            self.training = False
        else:
            raise ValueError("Mode can only be 'train' or 'test'")

# # Example usage
if __name__ == "__main__":
    # Simulated input data (batch of 2 samples, each with 5 features)
    np.random.seed(42)
    input_data = np.random.randn(2, 5)

    # Dropout layer with 20% dropout rate
    dropout_layer = Dropout(0.2)

    # Set mode to 'train'
    dropout_layer.setMode('train')
    output_training = dropout_layer.forward(input_data)
    print("Output during training:")
    print(output_training)

    # Set mode to 'test'
    dropout_layer.setMode('test')
    output_testing = dropout_layer.forward(input_data)
    print("\nOutput during testing:")
    print(output_testing)