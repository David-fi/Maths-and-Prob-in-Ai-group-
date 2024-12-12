import numpy as np
import tensorflow as tf
import traceback
import matplotlib.pyplot as plt

from NN import NeuralNetwork
from ActivationFunction import ActivationFunction
from SoftmaxLayer import SoftmaxLayer
from Dropout import Dropout

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
        # CIFAR-10 specific parameters
        self.input_size = 3072  # 32x32x3 image flattened
        self.num_classes = 10
        
        # Neural Network parameters
        self.no_of_hidden_layers = no_of_hidden_layers
        self.no_of_hidden_units = no_of_hidden_units
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.epochs = epochs
        
        # Neural Network instance
        self.network = None
        
        # Dataset
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_and_preprocess_data(self):
        """
        Load CIFAR-10 dataset and preprocess it
        """
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # Flatten images and normalize
        X_train = X_train.reshape(X_train.shape[0], -1).astype('float32') / 255.0
        X_test = X_test.reshape(X_test.shape[0], -1).astype('float32') / 255.0

        # One-hot encode labels
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
    
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
            print(traceback.format_exc())
    
    def train_network(self, activation_function):
        """
        Train the neural network on CIFAR-10 dataset
        Args:
            activation_function (String): Type of activation function.
        """
        print(f"Training Neural Network with {activation_function}...")
        try:
            if self.network is None:
                raise ValueError("Neural network not initialized. Call create_neural_network() first.")
            
            if self.X_train is None:
                raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")
            
            # Use data without transposing
            X_train_processed = self.X_train
            
            # Train the network
            print(f"Starting training for {self.epochs} epochs...")
            self.network.train(X_train_processed, self.y_train, activation_function, self.epochs)
            print("Training completed")
        
        except Exception as e:
            print("Error during training:")
            print(traceback.format_exc())

    
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
                raise ValueError("Neural network not trained. Call train_network() first.")
            
            # Predict on test data
            predictions = []
            for sample in self.X_test:
                # Run each sample through the network
                pred = self.network.run(sample.T, activation_function)
                predictions.append(np.argmax(pred))
            
            # Convert predictions and true labels to numpy arrays
            predictions = np.array(predictions)
            true_labels = np.argmax(self.y_test, axis=1)
            
            # Calculate accuracy
            accuracy = np.mean(predictions == true_labels)
            print(f"Network Accuracy: {accuracy * 100:.2f}%")
            #self.network.plot_loss()
            return accuracy
        
        except Exception as e:
            print("Error during evaluation:")
            print(traceback.format_exc())
            return None
    
    def run(self, activation_function):
        """
        Run the entire CIFAR-10 dataset
        Args:
            activation_function (String): Type of activation function.
        """
        validActivationFunctions = ['sigmoidForward', 'reluForward']
        if activation_function not in validActivationFunctions:
                raise ValueError(f"{activation_function} is not a valid activation function.")
        print("Starting CIFAR-10 Experiment...")
        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Create neural network
        self.create_neural_network()
        
        # Train network
        self.train_network(activation_function)
        
        # Evaluate network
        accuracy = self.evaluate_network(activation_function)

# Main execution
if __name__ == "__main__":
    runner = CIFAR10Runner(
        no_of_hidden_layers = 3,
        no_of_hidden_units = [512, 128, 64],
        learning_rate = 0.01,
        dropout_rate = 0.2,
        epochs = 5  # Temporary reduced for quicker testing
    )
    
    try:
        runner.run('sigmoidForward')
    except Exception as e:
        print("Error in main execution:")
        print(traceback.format_exc())