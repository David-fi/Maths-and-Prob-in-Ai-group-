import numpy as np
import matplotlib.pyplot as plt

from ActivationFunction import ActivationFunction
from Dropout import Dropout
from SoftmaxLayer import SoftmaxLayer

class NeuralNetwork:
    def __init__(self, activationFunction, input_size, output_size, hidden_units, learning_rate, dropout_rate=0.5):

        # Parameters and hyperparameters initialisation 
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate

        self.weights = []
        self.biases = []
        self.velocities_weights = []
        self.velocities_biases = []
        self.dropout_layers = []
        self.loss_values = []

        self.activationFunction = ActivationFunction(activationFunction)

        # Initialize weights, biases, and dropout layers for each layer in the network
        layer_sizes = [input_size] + hidden_units + [output_size]
        for i in range(len(layer_sizes) - 1):

            # He initialization for weights: scaled random initialization for better convergence
            weights = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i])
            biases = np.zeros((1, layer_sizes[i + 1]))
            
            self.weights.append(weights)
            self.biases.append(biases)
            
            # Initialize momentum buffers to zeros for weights and biases
            self.velocities_weights.append(np.zeros_like(weights))
            self.velocities_biases.append(np.zeros_like(biases))

            # Add dropout to hidden layers only
            if i < len(layer_sizes) - 2:  
                self.dropout_layers.append(Dropout(self.dropout_rate))

    def forward(self, input_vector, training=True):
        """
        Perform forward propagation through the network.
        Args:
            input_vector: The input data (features).
            training: Boolean flag indicating whether the network is in training mode 
                    (to apply dropout) or evaluation mode (no dropout).
        Returns:
            output: The softmax probabilities for the output layer.
        """
        self.cache = {"A0": input_vector}
        
        # Forward pass through all hidden layers
        for i, (weight, bias) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            # Compute linear transformation Z = input data * weight + bias
            z = np.dot(self.cache[f"A{i}"], weight) + bias
            
            activation, cache = self.activationFunction.forward(z)
            self.cache[f"Z{i + 1}"] = cache
            
            # Apply dropout
            if training:
                activation = self.dropout_layers[i].forward(activation, training)
            self.cache[f"A{i + 1}"] = activation

        # Forward pass through the output layer
        # Compute logits for the output layer: Z_output = last hidden layer * weight + bias
        z_output = np.dot(self.cache[f"A{len(self.weights) - 1}"], self.weights[-1]) + self.biases[-1]
        output = SoftmaxLayer.softmaxForward(z_output)
        self.cache["Z_output"] = z_output
        return output

    def backward(self, forward_output, target_vector):
        """
        Perform backpropagation to compute gradients and update weights and biases.
        Args:
            forward_output: The predicted probabilities from the forward pass (softmax output).
            target_vector: The true labels in one-hot encoded format.
        """
        grads = {}
        dz_output = SoftmaxLayer.softmaxBackward(forward_output, target_vector)
        # Gradients for the output layer weights and biases
        grads[f"dW{len(self.weights) - 1}"] = np.dot(self.cache[f"A{len(self.weights) - 1}"].T, dz_output)
        grads[f"db{len(self.weights) - 1}"] = np.sum(dz_output, axis=0, keepdims=True)

        # Backpropagate the error to the previous layer
        dout = np.dot(dz_output, self.weights[-1].T)

        # Backpropagation through all hidden layers in reverse order (excluding output layer)
        for i in reversed(range(len(self.weights) - 1)):
            if i < len(self.dropout_layers):
                dout = self.dropout_layers[i].backward(dout)

            dz = self.activationFunction.backward(dout, self.cache[f"Z{i + 1}"])

            grads[f"dW{i}"] = np.dot(self.cache[f"A{i}"].T, dz)
            grads[f"db{i}"] = np.sum(dz, axis=0, keepdims=True)

            dout = np.dot(dz, self.weights[i].T)

        # Update weights and biases using momentum-based gradient descent
        for i in range(len(self.weights)):
            self.velocities_weights[i] = 0.9 * self.velocities_weights[i] + 0.1 * grads[f"dW{i}"]
            self.velocities_biases[i] = 0.9 * self.velocities_biases[i] + 0.1 * grads[f"db{i}"]
            
            self.weights[i] -= self.learning_rate * self.velocities_weights[i]
            self.biases[i] -= self.learning_rate * self.velocities_biases[i]

    def train(self, input_vector, target_vector, x_val, y_val, epochs, batch_size):
        """
        Train the neural network using mini-batch gradient descent.
        Args:
            input_vector: Training input data.
            target_vector: Training labels (one-hot encoded).
            x_val: Validation input data.
            y_val: Validation labels (one-hot encoded).
            epochs: Number of training epochs.
            batch_size: Size of each mini-batch.
        """
        self.val_losses = []  # List to store validation loss per epoch
        self.val_accuracies = []  # List to store validation accuracy per epoch

        for epoch in range(epochs):
            # Shuffle the training data at the start of each epoch
            perm = np.random.permutation(input_vector.shape[0])
            input_vector, target_vector = input_vector[perm], target_vector[perm]

            epoch_loss = 0
            for i in range(0, input_vector.shape[0], batch_size):
                x_batch = input_vector[i:i + batch_size]
                y_batch = target_vector[i:i + batch_size]

                output = self.forward(x_batch, training=True)
                self.backward(output, y_batch)

                batch_loss = -np.mean(np.sum(y_batch * np.log(output + 1e-8), axis=1))
                epoch_loss += batch_loss * x_batch.shape[0]

            epoch_loss /= input_vector.shape[0]
            self.loss_values.append(epoch_loss)

            # Validation loss and accuracy
            val_accuracy, val_loss = self.run(x_val, y_val, return_loss=True)

            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%")


    def run(self, input_data, true_labels, return_loss=False):
        """
        Evaluate the neural network on a dataset and optionally compute loss.
        
        Args:
            input_data: Input data (features), shape (num_samples, num_features).
            true_labels: True labels in one-hot encoded format, shape (num_samples, num_classes).
            return_loss: Boolean, whether to return the loss.
        
        Returns:
            accuracy: The accuracy of the network on the given dataset as a float value.
            loss (optional): The loss on the given dataset.
        """
        output = self.forward(input_data, training=False)
        predictions = np.argmax(output, axis=1)
        labels = np.argmax(true_labels, axis=1)
        
        accuracy = np.mean(predictions == labels)
        
        if return_loss:
            loss = -np.mean(np.sum(true_labels * np.log(output + 1e-8), axis=1))
            return accuracy, loss
        return accuracy

    def plot_loss(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_values, label='Training Loss', marker='o')
        plt.plot(self.val_losses, label='Validation Loss', marker='x')
        plt.title("Training and Validation Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(self.val_accuracies, label='Validation Accuracy', marker='x', color='green')
        plt.title("Validation Accuracy Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()
        plt.show()