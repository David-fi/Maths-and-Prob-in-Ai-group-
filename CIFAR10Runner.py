import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNetwork

class CIFAR10Runner:
    def __init__(self, activationFunction, hidden_units, learning_rate, epochs, batch_size, dropout_rate):
        # CIFAR-10 specific parameters
        self.input_size = 32 * 32 * 3
        self.output_size = 10

        # Neural Network parameters
        self.activationFunction = activationFunction
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate

    def load_data(self):
        """
        Load CIFAR-10 dataset and preprocess it.
        """
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # Flatten images and normalize
        x_train = x_train.reshape(x_train.shape[0], -1).astype("float32") / 255.0
        x_test = x_test.reshape(x_test.shape[0], -1).astype("float32") / 255.0

        # One-hot encode labels
        y_train = tf.keras.utils.to_categorical(y_train, self.output_size)
        y_test = tf.keras.utils.to_categorical(y_test, self.output_size)
        
        # Concatenate the training and test datasets features and labels 
        combined_train = np.concatenate([x_train, x_test], axis=0) 
        combined_test = np.concatenate([y_train, y_test], axis=0)  

        # Split the combined data into train, validate and test sets, in a randomized manner but with a fixed seed to ensure reproducibility of the results
        x_train_new, x_test, y_train_new, y_test = train_test_split(combined_train, combined_test, test_size=0.10, random_state=42, stratify=combined_test) 
      
        x_train, x_val, y_train, y_val = train_test_split(x_train_new, y_train_new, test_size=0.162, random_state=42, stratify=y_train_new)
        return x_train, y_train, x_val, y_val, x_test, y_test

    def run(self):
        """
        Train and evaluate the Neural Network.
        """
        print("Loading CIFAR-10 data...")
        x_train, y_train, x_val, y_val, x_test, y_test = self.load_data()

        print("Initializing the Neural Network...")
        network = NeuralNetwork(self.activationFunction, self.input_size, self.output_size, self.hidden_units, self.learning_rate, self.dropout_rate)

        print("Training the Neural Network...")
        network.train(x_train, y_train, x_val, y_val, self.epochs, self.batch_size)

        #print("Final Evaluation on Test Set...")
        #test_accuracy = network.run(x_test, y_test)
        #print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

        network.plot_loss()

if __name__ == "__main__":
    runner = CIFAR10Runner(
        activationFunction = "relu",
        hidden_units = [1024, 512, 256],
        learning_rate = 0.001,
        epochs = 5,
        batch_size = 64,
        dropout_rate = 0.2
    )
    runner.run()
