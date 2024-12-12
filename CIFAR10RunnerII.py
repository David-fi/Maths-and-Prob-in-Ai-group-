import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from NeuralNetworkIII import NeuralNetwork
from ActivationFunction import ActivationFunction
from SoftmaxLayer import SoftmaxLayer
from Dropout import Dropout
from keras.src.callbacks import TensorBoard

class CIFAR10Runner:
    """
    Author: Abdelrahmane Bekhli
    Date: 2024-10-12
    Description: This class performs dropouts.
    """
    def __init__(self, no_of_hidden_layers, no_of_hidden_units, learning_rate, dropout_rate, epochs):
        """
        Initialize CIFAR-10 Runner with Neural Network configuration
        """
        print("Initializing CIFAR-10 Runner...")
        self.input_size = 3072  # 32x32x3 image flattened
        self.num_classes = 10

        self.no_of_hidden_layers = no_of_hidden_layers
        self.no_of_hidden_units = no_of_hidden_units
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.epochs = epochs

        self.network = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_and_preprocess_data(self):
        """
        Load CIFAR-10 dataset and preprocess it
        """
        print("Loading and preprocessing CIFAR-10 dataset...")
        try:
            (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

            X_train = X_train.astype('float32') / 255.0
            X_test = X_test.astype('float32') / 255.0

            y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
            y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

            self.X_train = X_train
            self.X_test = X_test
            self.y_train = y_train
            self.y_test = y_test

            print(f"Training data shape: {self.X_train.shape}")
            print(f"Training labels shape: {self.y_train.shape}")
            print(f"Test data shape: {self.X_test.shape}")
            print(f"Test labels shape: {self.y_test.shape}")

        except Exception as e:
            print("Error loading dataset:")
            print(e)

    def create_neural_network(self):
        """
        Create and configure the neural network
        """
        print("Creating Neural Network...")
        try:
            self.network = NeuralNetwork(
                self.no_of_hidden_layers, 
                self.input_size, 
                self.num_classes, 
                self.no_of_hidden_units,
                self.learning_rate,
                self.dropout_rate,
                ActivationFunction,
                SoftmaxLayer,
                Dropout
            )
            print("Neural Network created successfully")
        except Exception as e:
            print("Error creating neural network:")
            print(e)

    def train_network_with_tensorboard(self, activation_function):
        """
        Train the neural network on CIFAR-10 dataset with TensorBoard support
        Args:
            activation_function (String): Type of activation function.
        """
        print(f"Training Neural Network with {activation_function}...")
        try:
            if self.network is None:
                raise ValueError("Neural network not initialized. Call create_neural_network() first.")

            if self.X_train is None:
                raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")

            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_summary_writer = tf.summary.create_file_writer(log_dir)

            self.network.loss_values = []  # Initialize list to store loss values

            for epoch in range(self.epochs):
                output = self.network.forward(self.X_train, activation_function)

                loss = -np.mean(np.sum(self.y_train * np.log(output), axis=1))
                self.network.loss_values.append(loss)  # Store loss values for plotting

                self.network.backward(self.y_train, activation_function)

                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=epoch)

                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss}")

        except Exception as e:
            print("Error during training:")
            print(e)

    def evaluate_network(self, activation_function):
        """
        Evaluate the trained network on test data
        Args:
            activation_function (String): Type of activation function.
        Returns:
            float: accuracy of the network.
        """
        print("Evaluating Neural Network...")
        try:
            if self.network is None:
                raise ValueError("Neural network not trained. Call train_network_with_tensorboard() first.")

            predictions = []
            for sample in self.X_test:
                pred = self.network.run(sample.T, activation_function)
                predictions.append(np.argmax(pred))

            predictions = np.array(predictions)
            true_labels = np.argmax(self.y_test, axis=1)

            accuracy = np.mean(predictions == true_labels)
            print(f"Network Accuracy: {accuracy * 100:.2f}%")
            self.plot_loss()
            return accuracy

        except Exception as e:
            print("Error during evaluation:")
            print(e)
            return None

    def plot_loss(self):
        """
        Plot the training loss over epochs
        """
        try:
            plt.figure(figsize=(8, 6))
            plt.plot(range(1, len(self.network.loss_values) + 1), self.network.loss_values, marker='o', linestyle='-', label='Training Loss')
            plt.title('Training Loss over Epochs', fontsize=14)
            plt.xlabel('Epochs', fontsize=12)
            plt.ylabel('Loss', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            plt.show()
        except Exception as e:
            print("Error plotting loss:")
            print(e)

    def run_with_tensorboard(self, activation_function):
        """
        Run the entire CIFAR-10 dataset with TensorBoard integration
        Args:
            activation_function (String): Type of activation function.
        """
        validActivationFunctions = ['sigmoidForward', 'reluForward']
        if activation_function not in validActivationFunctions:
            raise ValueError(f"{activation_function} is not a valid activation function.")

        print("Starting CIFAR-10 Experiment...")
        self.load_and_preprocess_data()
        self.create_neural_network()
        self.train_network_with_tensorboard(activation_function)
        self.evaluate_network(activation_function)

# Main execution
if __name__ == "__main__":
    runner = CIFAR10Runner(
        no_of_hidden_layers = 3,
        no_of_hidden_units = [128, 128, 128],
        learning_rate = 0.01,
        dropout_rate = 0.2,
        epochs = 5  # Increased for better monitoring
    )

    try:
        runner.run_with_tensorboard('reluForward')
    except Exception as e:
        print("Error in main execution:")
        print(e)


#tensorboard --logdir="logs\fit"
