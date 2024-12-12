import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, num_layers, input_size, output_size, hidden_units, learning_rate, dropout_rate, activation_fn, softmax_layer, dropout_layer):
        """
        Initialize the Neural Network.

        Args:
            num_layers (int): Number of layers (excluding input and output layers).
            input_size (int): Number of input features.
            output_size (int): Number of output classes.
            hidden_units (list): List of hidden units per layer.
            learning_rate (float): Learning rate for gradient descent.
            dropout_rate (float): Dropout probability.
            activation_fn (class): ActivationFunction class reference.
            softmax_layer (class): SoftmaxLayer class reference.
            dropout_layer (class): Dropout class reference.
        """
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.activation_fn = activation_fn
        self.softmax_layer = softmax_layer()
        self.dropout_layer = dropout_layer(dropoutRate=dropout_rate, training=True)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.cache = {}
        
        layer_sizes = [input_size] + hidden_units + [output_size]
        for i in range(len(layer_sizes) - 1):
            # self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01)
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
    
    def forward(self, X, activation_function):
        """
        Perform forward propagation.

        Args:
            X (np.array): Input data.
            activation_function (str): Activation function name.

        Returns:
            np.array: Output of the network.
        """
        self.cache["A0"] = X
        A = X
        
        for i in range(self.num_layers):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            A, cache = self.activation_fn.whichActivationFunctionForwardPass(activation_function, Z)
            if A is None or cache is None:
                raise ValueError(f"Forward pass returned None for activations at layer {i}")
            self.cache[f"A{i + 1}"] = A
            self.cache[f"Z{i + 1}"] = cache

            
            # Apply dropout
            if i < self.num_layers - 1:  # Don't apply dropout on the last layer
                A = self.dropout_layer.dropoutForward(A)
        
        # Output layer
        Z_output = np.dot(A, self.weights[-1]) + self.biases[-1]
        output = self.softmax_layer.softmaxForward(Z_output)
        self.cache["Z_output"] = Z_output
        return output
    
    def backward(self, Y, activation_function):
        """
        Perform backward propagation.

        Args:
            Y (np.array): True labels.
            activation_function (str): Activation function name.
        """
        grads = {}
        m = Y.shape[0]
        
        # Output layer gradient
        dZ_output = self.softmax_layer.softmaxBackward(Y)
        grads[f"dW{self.num_layers}"] = np.dot(self.cache[f"A{self.num_layers - 1}"].T, dZ_output)
        grads[f"db{self.num_layers}"] = np.sum(dZ_output, axis=0, keepdims=True)
        
        dA_prev = np.dot(dZ_output, self.weights[-1].T)
        
        for i in reversed(range(self.num_layers)):
            # Map forward activation to backward activation
            if activation_function == "sigmoidForward":
                activation_function_backward = "sigmoidBackward"
            elif activation_function == "reluForward":
                activation_function_backward = "reluBackward"
            else:
                raise ValueError(f"Unsupported activation function: {activation_function}")

            # Dropout gradient
            if i < self.num_layers - 1:  # Apply dropout only to hidden layers
                dA_prev = self.dropout_layer.dropoutBackward(dA_prev)
            
            # Backward activation
            dZ = self.activation_fn.whichActivationFunctionBackwardPass(
                activation_function_backward, dA_prev, self.cache[f"Z{i + 1}"]
            )
            grads[f"dW{i}"] = np.dot(self.cache[f"A{i}"].T, dZ)
            grads[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True)
            
            if i > 0:
                dA_prev = np.dot(dZ, self.weights[i].T)
        
        # Update weights and biases
        for i in range(self.num_layers):
            self.weights[i] -= self.learning_rate * grads[f"dW{i}"]
            self.biases[i] -= self.learning_rate * grads[f"db{i}"]
 
    def train(self, X, Y, activation_function, epochs):
        """
        Train the Neural Network.

        Args:
            X (np.array): Training input data.
            Y (np.array): Training labels.
            activation_function (str): Activation function name.
            epochs (int): Number of epochs.
        """
        self.loss_values = []  # Initialize list to store loss values

        for epoch in range(epochs):

            output = self.forward(X, activation_function)
            
            # Calculate loss
            loss = -np.mean(np.sum(Y * np.log(output), axis=1))
            self.loss_values.append(loss)

            self.backward(Y, activation_function)

            #if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Loss: {loss}")

    def plot_loss(self):
        """
        Plot the training loss over epochs with epochs on the Y-axis and loss on the X-axis.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(self.loss_values, range(1, len(self.loss_values) + 1), marker='o', linestyle='-', label='Training Loss')
        plt.title('Training Loss over Epochs', fontsize=14)
        plt.ylabel('Epochs', fontsize=12)  # Y-axis is now epochs
        plt.xlabel('Loss', fontsize=12)    # X-axis is now loss
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.show()
    
    def run(self, X, activation_function):
        """
        Run the Neural Network for inference.

        Args:
            X (np.array): Input data.
            activation_function (str): Activation function name.

        Returns:
            np.array: Output predictions.
        """
        return self.forward(X, activation_function)
