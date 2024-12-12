import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm


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
        # TODO IMPLEMENT create_weight_matrices()
        layer_sizes = [input_size] + hidden_units + [output_size]
        for i in range(len(layer_sizes) - 1):
            #self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * 0.01)
            self.weights.append(np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i]))
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
    
    def truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

    def create_weight_matrices(self):
        """ A method to initialize the weight matrices of the neural network"""

        self.weights_in_hidden = np.zeros((self.no_of_in_units, self.no_of_hidden_units.size))
        self.weights_hidden_hidden = [] 
        self.weights_hidden_out = np.zeros((len(self.no_of_hidden_units), self.no_of_out_units))

        
        rad = 1 / np.sqrt(self.no_of_in_units)
        X = self.truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_units[0], 
                                        self.no_of_in_units))
        
        for i in range(0, len(self.no_of_hidden_units) - 1):
            
            rad = 1 / np.sqrt(self.no_of_hidden_units[i])
            X = self.truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
            self.weights_hidden_hidden_element = X.rvs((self.no_of_hidden_units[i+1], 
                                                        self.no_of_hidden_units[i]))
            self.weights_hidden_hidden.append(self.weights_hidden_hidden_element)
            
            print("self.weights_hidden_hidden[i]:", self.weights_hidden_hidden[i])
            print("Type of self.weights_hidden_hidden[i]:", type(self.weights_hidden_hidden[i]))
            
            
            print("self.weights_hidden_hidden:", self.weights_hidden_hidden)
            print("Type of self.weights_hidden_hidden:", type(self.weights_hidden_hidden))        
        
        rad = 1 / np.sqrt(self.no_of_hidden_units[len(self.no_of_hidden_units)-1])
        X = self.truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_units, 
                                        self.no_of_hidden_units[len(self.no_of_hidden_units)-1]))
        
            
        print("AFTER LOOP self.weights_hidden_hidden:", self.weights_hidden_hidden)
        print("AFTER LOOP Type of self.weights_hidden_hidden:", type(self.weights_hidden_hidden)) 



    def forward(self, input_vector, activation_function):
        """
        Perform forward propagation.
        """
        input_vector = np.atleast_2d(input_vector)  # Ensure input is 2D
        self.cache["A0"] = input_vector
        layer_input = input_vector

        for i in range(self.num_layers):
            # Compute forward propagation
            Z = np.dot(layer_input, self.weights[i]) + self.biases[i]
            layer_input, cache = self.activation_fn.whichActivationFunctionForwardPass(activation_function, Z)
            self.cache[f"A{i + 1}"] = layer_input
            self.cache[f"Z{i + 1}"] = cache

            # Apply dropout only on hidden layers
            if i < self.num_layers - 1:
                layer_input = self.dropout_layer.dropoutForward(layer_input)

        # Output layer
        Z_output = np.dot(layer_input, self.weights[-1]) + self.biases[-1]
        output = self.softmax_layer.softmaxForward(Z_output)
        self.cache["Z_output"] = Z_output
        return output

    def backward(self, target_vector, forward_output, activation_function):
        """
        Perform backward propagation.
        """
        grads = {}
        num_samples = target_vector.shape[0]

        # Output layer gradient
        dZ_output = self.softmax_layer.softmaxBackward(forward_output, target_vector)
        grads[f"dW{self.num_layers}"] = np.dot(self.cache[f"A{self.num_layers - 1}"].T, dZ_output)
        grads[f"db{self.num_layers}"] = np.sum(dZ_output, axis=0, keepdims=True)

        dout = np.dot(dZ_output, self.weights[-1].T)

        for i in reversed(range(self.num_layers)):
            # Backward activation
            backward_activation = f"{activation_function[:-7]}Backward"
            dZ = self.activation_fn.whichActivationFunctionBackwardPass(
                backward_activation, dout, self.cache[f"Z{i + 1}"]
            )

            # Gradients
            grads[f"dW{i}"] = np.dot(self.cache[f"A{i}"].T, dZ)
            grads[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True)

            # Apply dropout gradient
            if i > 0:
                dout = np.dot(dZ, self.weights[i].T)
                dout = self.dropout_layer.dropoutBackward(dout)

        # Update parameters
        for i in range(self.num_layers):
            self.weights[i] -= self.learning_rate * grads[f"dW{i}"]
            self.biases[i] -= self.learning_rate * grads[f"db{i}"]


    def train(self, input_vector, target_vector, activation_function, epochs):

        self.loss_values = []  # Initialize list to store loss values

        for epoch in range(epochs):

            forward_output = self.forward(input_vector, activation_function)
            
            # Calculate loss
            loss = -np.mean(np.sum(target_vector * np.log(forward_output), axis=1))
            self.loss_values.append(loss)

            self.backward(target_vector, forward_output, activation_function)

            #if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Loss: {loss}")

    def run(self, X, activation_function):
        """
        Run the Neural Network for inference.

        Args:
            X (np.array): Input data.
            activation_function (str): Activation function name.

        Returns:
            np.array: Output predictions.
        """
        self.dropout_layer.setMode(False)
        return self.forward(X, activation_function)
    