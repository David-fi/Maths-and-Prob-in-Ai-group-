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
        # Define hyperparameter options        
        # Create an empty list to store the configurations
        x_train, y_train, x_val, y_val, x_test, y_test = CIFAR10Runner.load_data()
        all_models = []
        
        # Unpack the values from the param_grid dictionary
        activationFunction = param_grid['activationFunction']
        hidden_units = param_grid['hidden_units']
        dropout_rate = param_grid['dropout_rate']
        epoch = param_grid['epoch']
        batch_size = param_grid['batch_size']
        l2_lambda = param_grid['l2_lambda']
        
        adamOptimiser = AdamOptimiser(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0.01)
        
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
                                    output_size = 10,
                                    hidden_units = [hu],
                                    dropout_rate = dr,
                                    optimisers = [adamOptimiser],
                                    epoch = ep,
                                    batch_size = bs,
                                    l2_lambda = ll
                                ) 

                                all_models.append(model)
                                

        print(f'Total configs: {len(all_models)}')
        
        scores = [(model.train(x_train, y_train, x_val, y_val, return_val_accuracy=False), model) for model in all_models]
        scores2 = [(model.run(x_val, y_val), model) for _, model in scores]

        best_score, best_model = max(scores2, key=lambda x: x[0]) 
        print(f"Best configuration: {best_model}", f"Print scores: {best_score}")


        print("List of models/different hyperparamenet combinations evaluated and their validation accuracy:")
        for score, model in scores2:
            print(f"Score: {score}, Model: {model}") 
    
        # Print the models list (all models generated during the search)
        print("\nList of models/different hyperparamenet combinations evaluated:")
        for model in all_models:
            print(model)
      
        
    
if __name__ == "__main__":

    
    param_grid = {
        'activationFunction': ['relu', 'tanh', 'sigmoid'],
        'hidden_units': [randint(10, 2000) for _ in range(randint(3, 10))],
        'dropout_rate': [0.1, 0.5],
        'epoch': [10, 100],
        'batch_size': [16, 128],
        'l2_lambda': [1e-6, 1e-4]
    }
    
    
    param_grid_for_quick_testing = {
        'activationFunction': ['relu'],
        'hidden_units': [randint(10, 20) for _ in range(randint(3, 5))],
        'dropout_rate': [0.1],
        'epoch': [3, 4, 5],
        'batch_size': [128],
        'l2_lambda': [1e-4]
    }
    
    adamOptimiser = AdamOptimiser(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay=0.01)
    sgdMomentumOptimiser = SGDMomentumOptimiser(learning_rate=0.001, momentum=0.9, decay=0.01)
    sgdOptimiser = SGDOptimiser(learning_rate=0.001, decay=0.01) 

    GridSearch(param_grid_for_quick_testing)

