import numpy as np
import matplotlib.pyplot as plt
import time, gc

from ActivationFunction import ActivationFunction
from Dropout import Dropout
from SoftmaxLayer import SoftmaxLayer
from BatchNormalisation import BatchNormalisation
from Optimisers import AdamOptimiser, SGDMomentumOptimiser

class NeuralNetwork:
    def __init__(self, activationFunction, input_size, output_size, hidden_units, optimiser_type, learning_rate, dropout_rate):
        print("Initializing the Neural Network...")

        # Parameters and hyperparameters initialisation 
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate

        self.weights = []
        self.biases = []
        self.dropout_layers = []
        self.batch_norm_layers = []
        self.loss_values = []

        self.activationFunction = ActivationFunction(activationFunction)

        # Initialize optimiser
        if optimiser_type == "adam":
            self.optimiser = AdamOptimiser(learning_rate)
        elif optimiser_type == "sgd_momentum":
            self.optimiser = SGDMomentumOptimiser(learning_rate)
        else:
            raise ValueError(f"Unsupported optimiser type: {optimiser_type}")

        # Initialize weights, biases, dropout, and batch normalization layers for each layer in the network
        layer_sizes = [input_size] + hidden_units + [output_size]
        for i in range(len(layer_sizes) - 1):

            # He initialization for weights
            weights = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2 / layer_sizes[i])
            biases = np.zeros((1, layer_sizes[i + 1]))
            
            self.weights.append(weights)
            self.biases.append(biases)

            # Add dropout to hidden layers only
            if i < len(layer_sizes) - 2:  
                self.dropout_layers.append(Dropout(self.dropout_rate))

            # Add batch normalization layers to hidden layers only
            if i < len(layer_sizes) - 2:
                self.batch_norm_layers.append(BatchNormalisation(layer_sizes[i + 1]))

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

            # Apply activation function
            activation, cache = self.activationFunction.forward(z)

            # Apply batch normalization
            if i < len(self.batch_norm_layers):
                activation = self.batch_norm_layers[i].forward(activation, training=training)

            # Apply dropout
            if training and i < len(self.dropout_layers):
                activation = self.dropout_layers[i].forward(activation, training=training)

            self.cache[f"Z{i + 1}"] = cache
            self.cache[f"A{i + 1}"] = activation

        # Forward pass through the output layer
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

            if i < len(self.batch_norm_layers):
                dout = self.batch_norm_layers[i].backward(dout)

            dz = self.activationFunction.backward(dout, self.cache[f"Z{i + 1}"])

            grads[f"dW{i}"] = np.dot(self.cache[f"A{i}"].T, dz)
            grads[f"db{i}"] = np.sum(dz, axis=0, keepdims=True)

            dout = np.dot(dz, self.weights[i].T)

        # Update weights and biases using the optimiser
        for i in range(len(self.weights)):
            self.weights[i] = self.optimiser.update_weights(self.weights[i], grads[f"dW{i}"])
            self.biases[i] = self.optimiser.update_weights(self.biases[i], grads[f"db{i}"])

    def train(self, input_vector, target_vector, x_val, y_val, epochs, batch_size):
        """
        Train the neural network on the provided dataset.

        Args:
            input_vector (numpy.ndarray): Training data features.
            target_vector (numpy.ndarray): Training data labels in one-hot encoded format.
            x_val (numpy.ndarray): Validation data features.
            y_val (numpy.ndarray): Validation data labels in one-hot encoded format.
            epochs (int): Number of training epochs.
            batch_size (int): Size of mini-batches for training.
        """
        # Log the size of the training dataset
        print(f"Training dataset size: {input_vector.shape[0]} samples")

        # Initialize lists to store validation losses and accuracies
        self.val_losses = []
        self.val_accuracies = []

        # Total number of samples in the training data
        num_samples = input_vector.shape[0]
        batch_indices = np.arange(0, num_samples, batch_size)
        print(f"Total batches per epoch: {len(batch_indices)}")

        print("Training the Neural Network...")
        for epoch in range(epochs):
            epoch_start = time.time()
            # Shuffle the dataset to ensure randomness in mini-batch selection
            perm = np.random.permutation(num_samples)
            input_vector, target_vector = input_vector[perm], target_vector[perm]

            # Tracks the total loss for the current epoch
            epoch_loss = 0
            # Tracks the total time spent on batches in the current epoch
            batch_time_total = 0

            for start_idx in batch_indices:
                # Track the start time of processing this batch
                batch_start = time.time()
                end_idx = min(start_idx + batch_size, num_samples)
                x_batch = input_vector[start_idx:end_idx]
                y_batch = target_vector[start_idx:end_idx]

                # Perform forward propagation to compute predictions for the batch
                output = self.forward(x_batch, training=True)
                self.backward(output, y_batch)

                # Compute the batch loss using cross-entropy
                batch_loss = -np.mean(np.sum(y_batch * np.log(output + 1e-8), axis=1))
                epoch_loss += batch_loss * x_batch.shape[0]

                batch_time_total += time.time() - batch_start

            epoch_loss /= num_samples
            self.loss_values.append(epoch_loss)

            val_start = time.time()
            
            val_accuracy, val_loss = self.run(x_val, y_val, return_loss=True)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            val_time = time.time() - val_start

            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%, "
                f"Batch Time: {batch_time_total:.2f}s, Val Time: {val_time:.2f}s, Total Time: {epoch_time:.2f}s")
            
            # clear memory
            del x_batch, y_batch, output
            gc.collect()

            # Early stopping condition: Stop if validation loss does not improve over the last 5 epochs
            if epoch > 10 and (self.val_losses[-1] > min(self.val_losses[-5:])):
                print(f"Early stopping at epoch {epoch + 1}")
                break


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
        plt.plot(self.loss_values, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss', color='orange')
        plt.title("Training and Validation Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize=(10, 6))
        plt.plot(self.val_accuracies, label='Validation Accuracy', color='green')
        plt.title("Validation Accuracy Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()
        plt.show()
