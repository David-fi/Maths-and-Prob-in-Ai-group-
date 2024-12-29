#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Paula

CREDITS/REFERENCES: Code for this class was partly based on https://github.com/akmuthun/Time-Series-Neural-Network-Grid-Search.git 
"""
import os
import random
from random import randint
import numpy as np
from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNetwork
from Optimisers import AdamOptimiser, SGDMomentumOptimiser, SGDOptimiser
from CIFAR10Runner import CIFAR10Runner
from concurrent.futures import ProcessPoolExecutor, as_completed

class GridSearch:
    def __init__(self, param_grid):
        # Ensure reproducibility of results
        np.random.seed(42)
        random.seed(42)

        # Load data
        x_train, y_train, x_val, y_val, x_test, y_test, output_size = CIFAR10Runner.load_data()

        # Unpack the values from the param_grid dictionary
        activationFunction = param_grid['activationFunction']
        hidden_units = param_grid['hidden_units']
        dropout_rate = param_grid['dropout_rate']
        epoch = param_grid['epoch']
        batch_size = param_grid['batch_size']
        l2_lambda = param_grid['l2_lambda']

        # Define optimizers
        best_adamOptimiser = AdamOptimiser(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0.01) #val Val 51.58%
        best_sgdMomentumOptimiser = SGDMomentumOptimiser(learning_rate=0.005, momentum=0.9, decay=0.01) # Val 52.20%
        best_sgdOptimiser = SGDOptimiser(learning_rate=0.01, decay=0.01) # Val 51.14% 

        # Generate configurations
        configurations = []
        for af in activationFunction:
            for hu in hidden_units:
                for dr in dropout_rate:
                    for ep in epoch:
                        for bs in batch_size:
                            for ll in l2_lambda:
                                config = {
                                    'activationFunction': af,
                                    'input_size': 32 * 32 * 3,
                                    'output_size': output_size,
                                    'hidden_units': hu,  # Pass the list of three hidden layer sizes
                                    'dropout_rate': dr,
                                    'optimisers': [best_sgdMomentumOptimiser],
                                    'epoch': ep,
                                    'batch_size': bs,
                                    'l2_lambda': ll
                                }
                                configurations.append(config)
                                print(f"Model: {config['activationFunction']}, Optimiser: {config['optimisers'][0].__class__.__name__}, Hidden Units: {config['hidden_units']}, Dropout Rate: {config['dropout_rate']}, Epoch: {config['epoch']}, Batch Size: {config['batch_size']}, L2 Lambda: {config['l2_lambda']}")

        print(f'Total configs: {len(configurations)}')

        # Train models in parallel
        results = []
        with ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(self.train_and_evaluate_model, config, x_train, y_train, x_val, y_val): config
                for config in configurations
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"An error occurred: {e}")

        # Log all evaluated models and their scores to a file
        with open("model_results.txt", "w") as log_file:
            for score, config in results:
                log_file.write(f"Model: {config['activationFunction']}, Optimiser: {config['optimisers'][0].__class__.__name__}, Hidden Units: {config['hidden_units']}, Dropout Rate: {config['dropout_rate']}, Epoch: {config['epoch']}, Batch Size: {config['batch_size']}, L2 Lambda: {config['l2_lambda']}, Accuracy: {score}\n")

        # Find the best model
        best_score, best_model_config = max(results, key=lambda x: x[0])
        print(f"Best configuration: {best_model_config['activationFunction']}, optimiser: {best_model_config['optimisers'][0].__class__.__name__}, hidden units: {best_model_config['hidden_units']}, dropout rate: {best_model_config['dropout_rate']}, epoch: {best_model_config['epoch']}, batch size: {best_model_config['batch_size']}, L2: {best_model_config['l2_lambda']}", f"Best accuracy: {best_score}")

        # Print all evaluated models and their scores
        print("\nList of models/different hyperparameter combinations evaluated and their validation accuracy:")
        for score, config in results:
            print(f"Accuracy: {score}, Config: {config['optimisers'][0].__class__.__name__}, Hidden Units: {config['hidden_units']}, Dropout Rate: {config['dropout_rate']}, Epoch: {config['epoch']}, Batch Size: {config['batch_size']}, L2 Lambda: {config['l2_lambda']}, Accuracy: {score}")

    def train_and_evaluate_model(self, config, x_train, y_train, x_val, y_val):
        model = NeuralNetwork(
            activationFunction=config['activationFunction'],
            input_size=config['input_size'],
            output_size=config['output_size'],
            hidden_units=config['hidden_units'],  # Pass the list of three hidden layer sizes
            dropout_rate=config['dropout_rate'],
            optimisers=config['optimisers'],
            epoch=config['epoch'],
            batch_size=config['batch_size'],
            l2_lambda=config['l2_lambda']
        )
        model.train(x_train, y_train, x_val, y_val)
        accuracy = model.run(x_val, y_val)
        return accuracy, config
        
    
if __name__ == "__main__":
    # Ensure reproducibility of results!!! Spec requirement. 
    np.random.seed(42)
    random.seed(42)
    
    param_grid1 = {
        'activationFunction': ['relu', 'tanh', 'sigmoid'],
        'hidden_units': [[randint(1024, 512), randint(512, 256), randint(256, 128)] for _ in range(3)],
        'dropout_rate': [0.1, 0.2, 0.5],
        'epoch': [10, 15, 30, 40],
        'batch_size': [16, 128],
        'l2_lambda': [1e-6, 1e-4]
    }
        
    param_grid3 = {
        'activationFunction': ['relu', 'tanh', 'sigmoid'],
        'hidden_units': [[randint(128, 512), randint(128, 512), randint(128, 512)] for _ in range(3)],  # Three hidden layers
        'dropout_rate': [0.1, 0.15, 0.2],
        'epoch': [30],
        'batch_size': [128],
        'l2_lambda': [1e-6, 1e-5, 1e-4]
    }

    param_grid4 = {
        'activationFunction': ['relu', 'tanh', 'sigmoid'],
        'hidden_units':[[randint(455, 505), randint(185, 235), randint(140, 190)] for _ in range(3)],
        'dropout_rate': [0.15, 0.2, 0.25],
        'epoch': [30],
        'batch_size': [128],
        'l2_lambda': [0.000005, 1e-5, 0.00005]
    
    }
    
    #GridSearch(param_grid_for_quick_testing)
    GridSearch(param_grid4)
