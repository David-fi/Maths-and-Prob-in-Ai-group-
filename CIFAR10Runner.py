import os
# disables oneDNN optimisations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
 
import random 
from random import randint
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNetwork
from Optimisers import AdamOptimiser, SGDMomentumOptimiser, SGDOptimiser
global model, runner
    
class CIFAR10Runner:
    def __init__(self, model): 
        # Neural Network parameters
        self.model = model
        
    @staticmethod
    def load_data():
        output_size = model_create().output_size
        
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # Flatten images and normalize
        x_train = x_train.reshape(x_train.shape[0], -1).astype("float32") / 255.0
        x_test = x_test.reshape(x_test.shape[0], -1).astype("float32") / 255.0

        # One-hot encode labels
        y_train = tf.keras.utils.to_categorical(y_train, output_size)
        y_test = tf.keras.utils.to_categorical(y_test, output_size)

        # Split the original training set into train/validation subsets
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size = 0.18, random_state = 42, stratify = y_train
        )
        print(f"Training dataset size: {x_train.shape[0]}")
        print(f"Validation dataset size: {x_val.shape[0]}")
        print(f"Test dataset size: {x_test.shape[0]}")
        
        # self.x_train = x_train
        # self.y_train = y_train
        # self.x_val = x_val
        # self.y_val = y_val
        # self.x_test = x_test
        # self.y_test = y_test
        
        return x_train, y_train, x_val, y_val, x_test, y_test

    def run(self):
        """
        Train and evaluate the Neural Network.
        """
        print("Loading CIFAR-10 data...")
        x_train, y_train, x_val, y_val, x_test, y_test = self.load_data()
        
        self.model.train(x_train, y_train, x_val, y_val, return_val_accuracy=False)

        # print("Final Evaluation on Test Set...")
        # test_accuracy = self.model.run(x_test, y_test)
        # print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
             
        self.model.plot_loss()
   
    
   
def model_create():  
    # Initialize optimizers
    adamOptimiser = AdamOptimiser(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0.01)
    sgdMomentumOptimiser = SGDMomentumOptimiser(learning_rate=0.001, momentum=0.9, decay=0.01)
    sgdOptimiser = SGDOptimiser(learning_rate=0.001, decay=0.01) 
    
    # Initialize the model
    model = NeuralNetwork(
        activationFunction="relu",
        input_size=32 * 32 * 3,
        output_size=10,
        hidden_units=[1024, 512, 256],
        dropout_rate=0.2,  # 0.4
        optimisers=[adamOptimiser],  # Use other optimizers if needed
        l2_lambda=0.0,
        epoch = 30,
        batch_size = 128
    )
    
    return model
    



if __name__ == "__main__":
    model = model_create()
    
    runner = CIFAR10Runner(
        model=model
    ) 
    
    # Run the model
    runner.run()

    