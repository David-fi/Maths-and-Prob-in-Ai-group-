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
from CIFAR10Runner import CIFAR10Runner


class GridSearch:
     def __init__(self, param_grid):
         # Ensure reproducibility of results!!! Spec requirement. 
        np.random.seed(42)
        random.seed(42)
         
        x_train, y_train, x_val, y_val, x_test, y_test, output_size = CIFAR10Runner.load_data()
        # Create an empty list to store the configurations
        all_models = []
        
        # Unpack the values from the param_grid dictionary
        activationFunction = param_grid['activationFunction']
        hidden_units = param_grid['hidden_units']
        dropout_rate = param_grid['dropout_rate']
        epoch = param_grid['epoch']
        batch_size = param_grid['batch_size']
        l2_lambda = param_grid['l2_lambda']
        
        best_adamOptimiser = AdamOptimiser(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0.01) 
        best_sgdMomentumOptimiser = SGDMomentumOptimiser(learning_rate=0.005, momentum=0.9, decay=0.01)
        best_sgdOptimiser = SGDOptimiser(learning_rate=0.01, decay=0.01)
        
        # Nested loops to generate the Cartesian product manually
        for af in activationFunction:
            for hu in hidden_units:
                for dr in dropout_rate:
                    for ep in epoch:
                        for bs in batch_size:
                            for ll in l2_lambda:

                                model = NeuralNetwork(
                                    activationFunction = af,
                                    input_size = 32 * 32 * 3,
                                    output_size = output_size,
                                    hidden_units = [hu],
                                    dropout_rate = dr,
                                    optimisers = [best_adamOptimiser],
                                    epoch = ep,
                                    batch_size = bs,
                                    l2_lambda = ll
                                ) 

                                all_models.append(model)
                                
                                
        print(f'Total configs: {len(all_models)}')
        
        scores = [(model.train(x_train, y_train, x_val, y_val, return_val_accuracy=False), model) for model in all_models]
        scores2 = [(model.run(x_val, y_val), model) for _, model in scores]

        best_score, best_model = max(scores2, key=lambda x: x[0]) 
        print(f"Best configuration: {best_model}", f"Print accuracy: {best_score}")


        print("List of models/different hyperparamenet combinations evaluated and their validation accuracy:")
        for score, model in scores2:
            print(f"Accuracy: {score}, Model: {model}") #Currently, FLAWED! MUST FIX!!! Cause now that return_val_accuracy=False, score in scores is not equal to val accuracy (I am not sure what is being printed)
    
        # Print the models list (all models generated during the search)
        print("\nList of models/different hyperparamenet combinations evaluated:")
        for model in all_models:
            print(model)

        
    
if __name__ == "__main__":
    # Ensure reproducibility of results!!! Spec requirement. 
    np.random.seed(42)
    random.seed(42)
    
    param_grid1 = {
        'activationFunction': ['relu', 'tanh', 'sigmoid'],
        'hidden_units': [randint(10, 2000) for _ in range(randint(3, 10))],
        'dropout_rate': [0.1, 0.2, 0.5],
        'epoch': [10, 15, 30, 40],
        'batch_size': [16, 128],
        'l2_lambda': [1e-6, 1e-4]
    }
    
    param_grid2 = {
        'activationFunction': ['relu', 'tanh', 'sigmoid'],
        'hidden_units': [randint(128, 256) for _ in range((3))],
        'dropout_rate': [0.1, 0.2, 0.5],
        'epoch': [30],
        'batch_size': [128],
        'l2_lambda': [1e-6, 1e-4]
    }
    
    param_grid_for_quick_testing = {
        'activationFunction': ['relu'],
        'hidden_units': [randint(10, 20) for _ in range(randint(3, 5))],
        'dropout_rate': [0.1],
        'epoch': [30],
        'batch_size': [128],
        'l2_lambda': [1e-4]
    }
    
    #GridSearch(param_grid_for_quick_testing)
    GridSearch(param_grid2)



