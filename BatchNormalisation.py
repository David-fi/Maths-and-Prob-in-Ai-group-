import numpy as np

class BatchNormalisation:
    def __init__(self, input_dim, epsilon=1e-5, momentum=0.9):
        """
        Initializes the Batch Normalization layer.

        Args:
            input_dim (int): The dimension of the input features.
            epsilon (float): A small number to avoid division by zero.
            momentum (float): Momentum for the running mean and variance.
        """
        self.gamma = np.ones((1, input_dim)) # Scale parameter
        self.beta = np.zeros((1, input_dim)) # Shift parameter
        self.epsilon = epsilon  
        self.momentum = momentum 

        # Running mean and variance (used during inference)
        self.running_mean = np.zeros((1, input_dim))
        self.running_var = np.ones((1, input_dim))

    def forward(self, x, training=True):
        """
        Forward pass for batch normalization.

        Args:
            x (numpy.ndarray): Input data of shape (batch_size, input_dim).
            training (bool): If True, updates the running mean/variance and normalizes using batch stats.

        Returns:
            numpy.ndarray: Batch-normalized output.
        """
        if training:
            # Calculate batch mean and variance
            batch_mean = np.mean(x, axis=0, keepdims=True) 
            batch_var = np.var(x, axis=0, keepdims=True) 

            # Precompute the inverse standard deviation for numerical efficiency
            self.inv_std = 1.0 / np.sqrt(batch_var + self.epsilon)

            # Normalize the batch: x_norm = (x - mean) / std
            self.x_norm = (x - batch_mean) * self.inv_std

            # Scale and shift using gamma and beta
            out = self.gamma * self.x_norm + self.beta

            # Update the running mean and variance for inference
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            # Cache values for the backward pass
            self.cache = (x, batch_mean, self.inv_std, batch_var)
            return out
        else:
            # Normalize using running mean and variance during inference
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            return self.gamma * x_norm + self.beta

    def backward(self, dout):
        """
        Backward pass for batch normalization.

        Args:
            dout (numpy.ndarray): Gradient of the loss with respect to the output, shape (batch_size, input_dim).

        Returns:
            numpy.ndarray: Gradient of the loss with respect to the input, shape (batch_size, input_dim).
        """
        # Unpack cached values from the forward pass
        x, mean, inv_std, var = self.cache
        m = x.shape[0]  # Batch size

        # Compute intermediate values
        x_mu = x - mean  # Deviation of input from the mean

        # Gradient of the normalized input with respect to the loss
        dx_norm = dout * self.gamma

        # Gradient with respect to variance
        dvar = -0.5 * np.sum(dx_norm * x_mu * inv_std**3, axis=0, keepdims=True)

        # Gradient with respect to mean
        dmean = (
            np.sum(-dx_norm * inv_std, axis=0, keepdims=True) +
            dvar * np.mean(-2.0 * x_mu, axis=0, keepdims=True)
        )

        # Gradient with respect to the input
        dx = dx_norm * inv_std + dvar * 2.0 * x_mu / m + dmean / m

        # Gradients with respect to the scale and shift parameters
        self.dgamma = np.sum(dout * self.x_norm, axis=0, keepdims=True) 
        self.dbeta = np.sum(dout, axis=0, keepdims=True)  

        return dx
