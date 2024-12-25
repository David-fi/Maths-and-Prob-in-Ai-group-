import os
# disables oneDNN optimisations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNetwork
from Optimisers import AdamOptimiser, SGDMomentumOptimiser, SGDOptimiser

class CIFAR10Runner:
    def __init__(self, activationFunction, hidden_units, epochs, batch_size, l2_reg, dropout_rate, initialOptimiser, secondaryOptimiser):
        # CIFAR-10 specific parameters
        self.input_size = 32 * 32 * 3
        self.output_size = 10

        # Neural Network parameters
        self.activationFunction = activationFunction
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.initialOptimiser = initialOptimiser
        self.secondaryOptimiser = secondaryOptimiser
        self.l2_reg = l2_reg

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # Flatten images and normalize
        x_train = x_train.reshape(x_train.shape[0], -1).astype("float32") / 255.0
        x_test = x_test.reshape(x_test.shape[0], -1).astype("float32") / 255.0

        # One-hot encode labels
        y_train = tf.keras.utils.to_categorical(y_train, self.output_size)
        y_test = tf.keras.utils.to_categorical(y_test, self.output_size)

        # Split the original training set into train/validation subsets
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size = 0.18, random_state = 42, stratify = y_train
        )
        print(f"Training dataset size: {x_train.shape[0]}")
        print(f"Validation dataset size: {x_val.shape[0]}")
        print(f"Test dataset size: {x_test.shape[0]}")
        return x_train, y_train, x_val, y_val, x_test, y_test

    def run(self):
        """
        Train and evaluate the Neural Network.
        """
        print("Loading CIFAR-10 data...")
        x_train, y_train, x_val, y_val, x_test, y_test = self.load_data()

        network = NeuralNetwork(self.activationFunction, self.input_size, self.output_size, self.hidden_units, self.dropout_rate, self.initialOptimiser, self.secondaryOptimiser, self.l2_reg)
        
        network.train(x_train, y_train, x_val, y_val, self.epochs, self.batch_size)

       # print("Final Evaluation on Test Set...")
       # test_accuracy = network.run(x_test, y_test)
       # print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        network.plot_loss()

if __name__ == "__main__":

    adamOptimiser = AdamOptimiser(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0.01)
    sgdMomentumOptimiser = SGDMomentumOptimiser(learning_rate=0.001, momentum=0.9, decay=0.01)
    sgdOptimiser = SGDOptimiser(learning_rate=0.001, decay=0.01)

    runner = CIFAR10Runner(
        activationFunction = "relu",
        hidden_units = [1024, 512, 256],
        epochs = 30,
        batch_size = 128,
        l2_reg = 0.0,
        dropout_rate = 0.2, # 0.4,
        initialOptimiser= adamOptimiser,
        secondaryOptimiser = sgdMomentumOptimiser
    ) 
    runner.run()
