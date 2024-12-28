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
        
        return x_train, y_train, x_val, y_val, x_test, y_test, output_size

    def run(self):
        """
        Train and evaluate the Neural Network.
        """
        print("Loading CIFAR-10 data...")
        x_train, y_train, x_val, y_val, x_test, y_test, output_size = self.load_data()
        
        self.model.train(x_train, y_train, x_val, y_val, return_val_accuracy=False)

        # print("Final Evaluation on Test Set...")
        # test_accuracy = self.model.run(x_test, y_test)
        # print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
             
        self.model.plot_loss()
   
    
   
def model_create():  
    # Initialize optimizers
    
    best_adamOptimiser = AdamOptimiser(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0.01) #val Val 51.58%, the better model cause Higher Validation Accuracy and Lower Validation Loss. Yes, greater gap but not yet concerning. so, tweak its other hyperparameters e.g. beta1=0.85 or beta2=0.99
   # Lesser optimizer/hyperparameter combinations 
    adamOptimiser2 = AdamOptimiser(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0.01) # Val 49.19%
    adamOptimiser3 = AdamOptimiser(learning_rate=0.0005, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0.01) # Val 49.06%
    
    best_sgdMomentumOptimiser = SGDMomentumOptimiser(learning_rate=0.005, momentum=0.9, decay=0.01) # Val 52.20%
    # Lesser optimizer/hyperparameter combinations 
    sgdMomentumOptimiser2 = SGDMomentumOptimiser(learning_rate=0.001, momentum=0.9, decay=0.01) # Val 51.23%
    sgdMomentumOptimiser3 = SGDMomentumOptimiser(learning_rate=0.01, momentum=0.9, decay=0.01) #Val 50.71%
    sgdMomentumOptimiser4 = SGDMomentumOptimiser(learning_rate=0.1, momentum=0.9, decay=0.01) # WORST ONE. Val 38.66% and very strange graph for loss. 
    sgdMomentumOptimiser5 = SGDMomentumOptimiser(learning_rate=0.0001, momentum=0.9, decay=0.01) #Val 43.50%

    best_sgdOptimiser = SGDOptimiser(learning_rate=0.01, decay=0.01) # Val 51.14% 
    # Lesser optimizer/hyperparameter combinations 
    sgdOptimiser2 = SGDOptimiser(learning_rate=0.001, decay=0.01) # Val 46.14%
    sgdOptimiser3 = SGDOptimiser(learning_rate=0.0001, decay=0.01) # Val 20% by epoch 11, I stopped the process 
    sgdOptimiser4 = SGDOptimiser(learning_rate=0.005, decay=0.01) # Val 49.99%
    
    
    # Initialize the model
    model = NeuralNetwork(
        activationFunction="relu", #VALUE TO BE CHANGED LATER
        input_size=32 * 32 * 3, #VALUE TO BE CHANGED LATER
        output_size=10, #VALUE TO BE CHANGED LATER
        hidden_units=[1024, 512, 256], #VALUE TO BE CHANGED LATER
        dropout_rate=0.2,  # 0.4 #VALUE TO BE CHANGED LATER
        optimisers=[sgdOptimiser4], #VALUE TO BE CHANGED LATER
        epoch = 30, #VALUE TO BE CHANGED LATER
        batch_size = 128, #VALUE TO BE CHANGED LATER
        l2_lambda=0.0 #VALUE TO BE CHANGED LATER
    )
    
    return model
    


if __name__ == "__main__":
    model = model_create()
    
    runner = CIFAR10Runner(
        model=model
    ) 
    
    # Run the model
    runner.run()
