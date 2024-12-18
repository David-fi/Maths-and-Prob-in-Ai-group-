import torch 
import matplotlib.pyplot as plt

from ActivationFunction import ActivationFunction
from Dropout import Dropout
from SoftmaxLayer import SoftmaxLayer

class NeuralNetwork:
    def __init__(self, activationFunction, input_size, output_size, hidden_units, learning_rate, dropout_rate=0.5):
        
        # Check if GPU is available to use
        cuda_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count()
        self.device = torch.device("cuda" if cuda_available else "cpu")
        if(cuda_available):
            print(f"Number of GPUs: {gpu_count}. using: {torch.cuda.get_device_name(0)}")
        else:
            print("No GPU detected, using CPU")

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

        # Initialize weights, biases, and dropout layers
        layer_sizes = [input_size] + hidden_units + [output_size]
        for i in range(len(layer_sizes) - 1):
            weights = torch.randn(layer_sizes[i], layer_sizes[i + 1], dtype = torch.float32, device=self.device) * torch.sqrt(torch.tensor(2.0 / layer_sizes[i]))
            biases = torch.zeros(1, layer_sizes[i + 1], dtype = torch.float32, device = self.device)

            self.weights.append(weights)
            self.biases.append(biases)

            self.velocities_weights.append(torch.zeros_like(self.weights[-1]))
            self.velocities_biases.append(torch.zeros_like(self.biases[-1]))

            if i < len(layer_sizes) - 2:
                self.dropout_layers.append(Dropout(self.dropout_rate))

    def forward(self, input_vector, training = True):
        # ensures input_vector is converted only if it's not already a tensor.
        if not isinstance(input_vector, torch.Tensor):
            input_vector = torch.tensor(input_vector, dtype = torch.float32, device = self.device)
        else:
            input_vector = input_vector.clone().detach().to(self.device).float()
        self.cache = {"A0": input_vector}

        # Forward pass through all hidden layers
        for i, (weight, bias) in enumerate(zip(self.weights[:-1], self.biases[:-1])):
            # Linear transformation
            z = torch.matmul(self.cache[f"A{i}"], weight) + bias
            self.cache[f"Z{i + 1}"] = z

            # Apply activation function
            activation, cache = self.activationFunction.forward(z)
            self.cache[f"A{i + 1}"] = activation
            self.cache[f"Cache{i + 1}"] = cache

            # Apply dropout
            if training:
                activation = self.dropout_layers[i].forward(activation, training)
                self.cache[f"A{i + 1}"] = activation

        # Forward pass through the output layer
        z_output = torch.matmul(self.cache[f"A{len(self.weights) - 1}"], self.weights[-1]) + self.biases[-1]
        output = SoftmaxLayer.softmaxForward(z_output)
        self.cache["Z_output"] = z_output
        return output


    def backward(self, forward_output, target_vector):
        grads = {}
        # ensures target_vector is converted only if it's not already a tensor.
        if not isinstance(target_vector, torch.Tensor):
            target_vector = torch.tensor(target_vector, dtype = torch.float32, device = self.device)
        else:
            target_vector = target_vector.clone().detach().to(self.device).float()

        # Compute gradient for output layer
        dz_output = SoftmaxLayer.softmaxBackward(forward_output, target_vector)

        grads[f"dW{len(self.weights) - 1}"] = torch.matmul(self.cache[f"A{len(self.weights) - 1}"].T, dz_output)
        grads[f"db{len(self.weights) - 1}"] = torch.sum(dz_output, dim = 0, keepdim = True)

        # Backpropagate the error to previous layer
        dout = torch.matmul(dz_output, self.weights[-1].T)

        # Backpropagation through all hidden layers in reverse order (excluding output layer)
        for i in reversed(range(len(self.weights) - 1)):
            if i < len(self.dropout_layers):
                dout = self.dropout_layers[i].backward(dout)

            dz = self.activationFunction.backward(dout, self.cache[f"Cache{i + 1}"])

            grads[f"dW{i}"] = torch.matmul(self.cache[f"A{i}"].T, dz)
            grads[f"db{i}"] = torch.sum(dz, dim = 0, keepdim = True)
            dout = torch.matmul(dz, self.weights[i].T)

        # Update weights and biases using momentum-based gradient descent
        for i in range(len(self.weights)):
            self.velocities_weights[i] = 0.9 * self.velocities_weights[i] + 0.1 * grads[f"dW{i}"]
            self.velocities_biases[i] = 0.9 * self.velocities_biases[i] + 0.1 * grads[f"db{i}"]

            self.weights[i] -= self.learning_rate * self.velocities_weights[i]
            self.biases[i] -= self.learning_rate * self.velocities_biases[i]


    def train(self, input_vector, target_vector, x_val, y_val, epochs, batch_size):
        self.val_losses = []
        self.val_accuracies = []

        for epoch in range(epochs):
            # Shuffle the training data at the start of each epoch
            perm = torch.randperm(input_vector.shape[0])
            input_vector, target_vector = input_vector[perm], target_vector[perm]

            epoch_loss = 0

            # Process the training data in mini-batches
            for i in range(0, input_vector.shape[0], batch_size):
                x_batch = input_vector[i:i + batch_size]
                y_batch = target_vector[i:i + batch_size]

                # convert to tensor
                x_batch = torch.tensor(x_batch, dtype = torch.float32, device = self.device)
                y_batch = torch.tensor(y_batch, dtype = torch.float32, device = self.device)

                output = self.forward(x_batch, training = True)

                batch_loss = -torch.mean(torch.sum(y_batch * torch.log(output + 1e-8), dim = 1))
                epoch_loss += batch_loss.item() * x_batch.shape[0]

                self.backward(output, y_batch)

            epoch_loss /= input_vector.shape[0]
            self.loss_values.append(epoch_loss)

            val_accuracy, val_loss = self.run(x_val, y_val, return_loss = True)

            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)

            print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy * 100:.2f}%")


    def run(self, input_data, true_labels, return_loss = False):
        """
        Evaluate the neural network on a dataset.
        Args:
            input_data: shape (num_samples, num_features).
            true_labels: True labels in one-hot encoded format.
            Optional return_loss: whether to return the loss value
        Returns:
            accuracy: The accuracy of the network on the given dataset as a float value.
        """
        # convert to tensor
        input_data = torch.tensor(input_data, dtype = torch.float32, device = self.device)
        true_labels = torch.tensor(true_labels, dtype = torch.float32, device = self.device)
        
        output = self.forward(input_data, training = False)
     
        predictions = torch.argmax(output, dim = 1) # Get the predicted class labels
        labels = torch.argmax(true_labels, dim = 1) # Get the true class labels

        accuracy = torch.mean((predictions == labels).float())

        if return_loss:
            loss = -torch.mean(torch.sum(true_labels * torch.log(output + 1e-8), dim = 1))
            return accuracy.item(), loss.item()
        return accuracy.item()


    def plot_loss(self):
        plt.figure(figsize = (10, 6))
        plt.plot(self.loss_values, label = 'Training Loss', marker = 'o')
        plt.plot(self.val_losses, label = 'Validation Loss', marker = 'x')
        plt.title("Training and Validation Loss Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()

        plt.figure(figsize = (10, 6))
        plt.plot(self.val_accuracies, label='Validation Accuracy', marker = 'x', color = 'green')
        plt.title("Validation Accuracy Over Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid()
        plt.show()