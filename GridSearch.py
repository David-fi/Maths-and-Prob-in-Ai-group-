#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Paula

CREDITS/REFERENCES: Code for this class was partly based on https://github.com/akmuthun/Time-Series-Neural-Network-Grid-Search.git 
"""
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

class GridSearch:
    def __init__(self, model, epochs, batch_size):
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Ensure reproducibility of results!!! Spec requirement. 
        np.random.seed(42)
        random.seed(42)
    
    def __repr__(self):
        return f"NeuralNetwork(activationFunction={self.activationFunction}, hidden_units={self.hidden_units}, " \
                f"dropout_rate={self.dropout_rate}, epochs={self.epochs}, batch_size={self.batch_size}, " \
                f"l2_lambda={self.l2_lambda})"
              

    def load_data(self):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

        # Flatten images and normalize
        x_train = x_train.reshape(x_train.shape[0], -1).astype("float32") / 255.0
        x_test = x_test.reshape(x_test.shape[0], -1).astype("float32") / 255.0

        # One-hot encode labels
        y_train = tf.keras.utils.to_categorical(y_train, self.model.output_size)
        y_test = tf.keras.utils.to_categorical(y_test, self.model.output_size)

        # Split the original training set into train/validation subsets
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size = 0.18, random_state = 42, stratify = y_train
        )
        print(f"Training dataset size: {x_train.shape[0]}")
        print(f"Validation dataset size: {x_val.shape[0]}")
        print(f"Test dataset size: {x_test.shape[0]}")
        return x_train, y_train, x_val, y_val, x_test, y_test
            

    def model_configs(self, param_grid):
        # Define hyperparameter options        
        # Create an empty list to store the configurations
        all_models = []
        
        # Unpack the values from the param_grid dictionary
        activationFunction = param_grid['activationFunction']
        hidden_units = param_grid['hidden_units']
        dropout_rate = param_grid['dropout_rate']
        l2_lambda = param_grid['l2_lambda']
        
        adamOptimiser = AdamOptimiser(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0.01)
        
        # Nested loops to generate the Cartesian product manually
        for af in activationFunction:
            for hu in hidden_units:
                for dr in dropout_rate:
                            for ll in l2_lambda:
                                # Create a configuration by combining each option
                              #  params1_for_NN_when_adam = [af, hu, dr, ep, bs, ll]
                                
                                model = NeuralNetwork(
                                    activationFunction = af,
                                    input_size = 32 * 32 * 3,
                                    output_size = 10,
                                    hidden_units = [hu],
                                    dropout_rate = dr,
                                    optimisers = [adamOptimiser],
                                    l2_lambda = ll
                                ) 

                                all_models.append(model)

        print(f'Total configs: {len(all_models)}')
        
        # Load the data (optional, depending on your implementation of train method in NeuralNetwork)
        x_train, y_train, x_val, y_val, x_test, y_test = self.load_data()
        
        scores = [(model.train(x_train, y_train, x_val, y_val, self.epochs, self.batch_size), model) for model in all_models]
        scores.sort(key=lambda x: x[0])  # Sort by score
        print(f"Best configuration: {scores[0][1]}", f"Print scores: {scores[0][0]}")
        return scores, all_models
    
        # print(f"Best configuration: {scores[0][1]}")
        # print(f"Best score: {scores[0][0]}")
        
        
    
if __name__ == "__main__":

    
    param_grid = {
        'activationFunction': ['relu', 'tanh', 'sigmoid'],
        'hidden_units': [randint(10, 2000) for _ in range(randint(3, 10))],
        'dropout_rate': [0.1, 0.5],
        'epochs': [10, 100],
        'batch_size': [16, 128],
        'l2_lambda': [1e-6, 1e-4]
    }
    
    
    param_grid_for_quick_testing = {
        'activationFunction': ['relu', 'tanh'],
        'hidden_units': [randint(10, 20) for _ in range(randint(3, 5))],
        'dropout_rate': [0.1],
        'epochs': [30],
        'batch_size': [128],
        'l2_lambda': [1e-4]
    }
    
    adamOptimiser = AdamOptimiser(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0.01)
    sgdMomentumOptimiser = SGDMomentumOptimiser(learning_rate=0.001, momentum=0.9, decay=0.01)
    sgdOptimiser = SGDOptimiser(learning_rate=0.001, decay=0.01) 

    modelImp = NeuralNetwork(
        activationFunction = "relu",
        input_size = 32 * 32 * 3,
        output_size = 10,
        hidden_units = [1024, 512, 256],
        dropout_rate = 0.2,  # 0.4,
        optimisers = [adamOptimiser], #, sgdMomentumOptimiser, sgdOptimiser],
        l2_lambda = 0.0
    )
    
    # Instantiate the GridSearch class
    grid_search = GridSearch(modelImp, epochs=30, batch_size=128)

    # Perform the grid search with the given parameter grid
    scores, all_models = grid_search.model_configs(param_grid_for_quick_testing)
    
    
# Print the scores list (sorted by performance)
print("Scores List (sorted by performance):")
for score, model in scores:
    print(f"Score: {score}, Model: {model}")

# Print the models list (all models generated during the search)
print("\nAll Models (unsorted):")
for model in all_models:
    print(model)
    
