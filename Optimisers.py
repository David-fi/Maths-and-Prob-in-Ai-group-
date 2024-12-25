import numpy as np

class Optimiser:
    def update_weights(self, weights, gradients, *args, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")

class AdamOptimiser(Optimiser):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize the Adam optimiser with hyperparameters:
        - learning_rate: Step size for parameter updates
        - beta1: Exponential decay rate for the first moment estimate
        - beta2: Exponential decay rate for the second moment estimate
        - epsilon: Small constant for numerical stability
        """
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # First moment vector (mean of gradients)
        self.v = {}  # Second moment vector (mean of squared gradients)
        self.t = 0   # Time step counter

    def update_weights(self, weights, gradients):
        """
        Update weights using the Adam optimisation algorithm.

        Args:
        - weights: Current weights of the layer
        - gradients: Gradients of the loss with respect to weights

        Returns:
        - Updated weights
        """
        if id(weights) not in self.m:
            # Initialize first and second moment vectors for the given weights
            self.m[id(weights)] = np.zeros_like(weights)
            self.v[id(weights)] = np.zeros_like(weights)

        # Increment time step
        self.t += 1

        # Update biased first and second moment estimate
        self.m[id(weights)] = self.beta1 * self.m[id(weights)] + (1 - self.beta1) * gradients
        self.v[id(weights)] = self.beta2 * self.v[id(weights)] + (1 - self.beta2) * (gradients ** 2)

        # Correct bias in first and second moment estimates
        m_hat = self.m[id(weights)] / (1 - self.beta1 ** self.t)
        v_hat = self.v[id(weights)] / (1 - self.beta2 ** self.t)

        # Update weights using the corrected moment estimates
        weights_update = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        weights -= weights_update

        return weights

class SGDMomentumOptimiser(Optimiser):
    def __init__(self, learning_rate=0.001, momentum=0.9):
        """
        Initialize the SGD optimiser with momentum:
        - learning_rate: Step size for parameter updates
        - momentum: Factor for exponential decay of the velocity term
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}

    def update_weights(self, weights, gradients):
        """
        Update weights using the SGD with Momentum algorithm.

        Args:
        - weights: Current weights of the layer
        - gradients: Gradients of the loss with respect to weights

        Returns:
        - Updated weights
        """
        # Use the id of weights to track unique velocity for each parameter
        if id(weights) not in self.velocity:
            self.velocity[id(weights)] = np.zeros_like(weights)

        # Update velocity using momentum and gradients
        self.velocity[id(weights)] = self.momentum * self.velocity[id(weights)] - self.learning_rate * gradients

        # Ensure the shapes are compatible for broadcasting
        self.velocity[id(weights)] = self.velocity[id(weights)].reshape(weights.shape)

        # Update weights using the velocity term
        weights += self.velocity[id(weights)]
        return weights

class SGDOptimiser(Optimiser):
    def __init__(self, learning_rate=0.001):
        """
        Initialize the SGD optimiser:
        - learning_rate: Step size for parameter updates
        """
        self.learning_rate = learning_rate

    def update_weights(self, weights, gradients):
        """
        Update weights using the basic Stochastic Gradient Descent algorithm.

        Args:
        - weights: Current weights of the layer
        - gradients: Gradients of the loss with respect to weights

        Returns:
        - Updated weights
        """
        weights -= self.learning_rate * gradients
        return weights
