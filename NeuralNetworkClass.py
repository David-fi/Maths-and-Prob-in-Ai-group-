#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 19:23:37 2024

@author: Paula Suarez Rodriguez
"""
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import List

import numpy as np

from ActivationFunction import ActivationFunction
from SoftmaxLayer import SoftmaxLayer
from Dropout import Dropout

from scipy.stats import truncnorm


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
        (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

class NeuralNetwork:
    
    #Constructor for Neural Network class. Allows for Number of hidden layers, 
    #units and activation functions to be changed (as specified in the cw) by 
    # passing these as parameters to the constructor
    
    def __init__(self, 
                 no_of_hidden_layers,
                 no_of_in_units, 
                 no_of_out_units, 
                 no_of_hidden_units: List[int], #List where each element is an integer representing the number of units/nodes in each hidden layer
                 learning_rate,
                 dropoutRate,
                 activation_function_class = ActivationFunction,
                 softmax_layer_class = SoftmaxLayer,
                 dropout_class = Dropout): # when methods within the Dropout class are called, 
                                           # we will be able to specify the dropout rate, 
                                           # as the class Dropout takes dropoutRate as a 
                                           # parameter in the constructor
                                           
        self.no_of_hidden_layers = no_of_hidden_layers
        self.no_of_in_units = no_of_in_units
        self.no_of_out_units = no_of_out_units
        self.no_of_hidden_units = no_of_hidden_units
        self.learning_rate = learning_rate 
        self.dropoutRate = dropoutRate
        self.activation_function_class = activation_function_class
        self.softmax_layer_class = softmax_layer_class
        self.dropout_class = dropout_class(dropoutRate=dropoutRate, mask=None, training=True)
        self.create_weight_matrices(no_of_hidden_units)
    
    
        
    def create_weight_matrices(self, no_of_hidden_units):
        """ A method to initialize the weight matrices of the neural network"""
        
        self.weights_hidden_hidden = [None]*len(no_of_hidden_units)
        
        rad = 1 / np.sqrt(self.no_of_in_units)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_in_hidden = X.rvs((self.no_of_hidden_units[0], 
                                       self.no_of_in_units))
        
        for i in range(0, len(no_of_hidden_units) - 1):
            rad = 1 / np.sqrt(self.no_of_hidden_units[i])
            X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
            self.weights_hidden_hidden[i] = X.rvs((self.no_of_hidden_units[i], 
                                                self.no_of_hidden_units[i+1]))
        
        rad = 1 / np.sqrt(self.no_of_hidden_units[len(no_of_hidden_units)-1])
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weights_hidden_out = X.rvs((self.no_of_out_units, 
                                        self.no_of_hidden_units[len(no_of_hidden_units)-1]))
        
    
    
    
    def train(self, input_vector, target_vector, activation_function_string, epochs=100):
        # input_vector and target_vector can be tuple, list or ndarray
        
        input_vector = np.array(input_vector).T # tuple
        target_vector = np.array(target_vector).T
        output_vector3 = [None]*len(self.weights_in_hidden)
        output_vector_hidden_hidden = [None]*len(output_vector3)
        dropout_output_vector_hidden_hidden = [None]*len(output_vector_hidden_hidden)
        dropout_output_vector_hidden_hidden_derivate = [None]*len(output_vector_hidden_hidden)
        tmp2hh = [None]*len(output_vector_hidden_hidden)
        hidden_errors = [None]*len(self.weights_hidden_hidden)
        tmp3hh = [None]*len(self.weights_hidden_hidden)
        weight_update_hidden_hidden = [None]*len(self.weights_hidden_hidden)
      #  Dropout.training = True
    
        for i in range(epochs): # loop allows to perform the following caclulation for every layer of the NN
        # i.e. performs a forward pass where the weighted sums and activation functions (sigmoidForward or ReLu) are
        # calculated and a backward pass where weighted updates are calculated using the derivative of the activation
        # function (wether that is sigmoidBackward or reluBackward)
        
        #FORWARD PASS:
            # calculates weighted sum of input to first hidden layer
            input_vector = input_vector.T
            output_vector1 = np.dot(self.weights_in_hidden, input_vector) #self.weights_in_hidden is a 2D array. input_vector is also a 2D array. So, their np.dot product, which is a matrix multiplication, will also be 2D array. Must DOUBLE-CHECK!
           
            # calculates activation function of input to first hidden layer
            output_vector_hidden = self.activation_function_class.whichActivationFunctionForwardPass(activation_function_string, output_vector1) # Choose between using activation function sigmoid or ReLu during the forward pass
           
            # calculate dropout of activation function of input to first hidden layer –– i.e. pass output_vector_hidden as input to droput layer. IMPORTANT! 
            self.dropout_class.setMode('train') 
            dropout_output_vector_hidden = self.dropout_class.dropoutForward(output_vector_hidden)
            
                        
            
            
            
            for j in range(0, len(self.weights_hidden_hidden)):
                #calculates weighted sum of first hidden layer to last hidden layer
                output_vector3[j] = np.dot(self.weights_hidden_hidden[j], dropout_output_vector_hidden) #self.weights_in_hidden is a 2D array. input_vector is also a 2D array. So, their np.dot product, which is a matrix multiplication, will also be 2D array. Must DOUBLE-CHECK!
                
                # calculates activation function of first hidden layer to last hidden layer
                output_vector_hidden_hidden[j] = self.activation_function_class.whichActivationFunctionForwardPass(activation_function_string, output_vector3[j]) # Choose between using activation function sigmoid or ReLu during the forward pass
                
                # calculate dropout of activation function of hidden layers –– i.e. pass output_vector_hidden_hidden as input to droput layer. IMPORTANT!
                self.dropout_class.setMode('train') 
                dropout_output_vector_hidden_hidden[j] = self.dropout_class.dropoutForward(output_vector_hidden_hidden[j])
             
            '''
            for j in range(len(self.weights_hidden_hidden)):
                #calculates weighted sum of first hidden layer to last hidden layer
                output_vector3[0] = np.dot(self.weights_hidden_hidden[0], dropout_output_vector_hidden) #self.weights_in_hidden is a 2D array. input_vector is also a 2D array. So, their np.dot product, which is a matrix multiplication, will also be 2D array. Must DOUBLE-CHECK!
                
                #calculates weighted sum of first hidden layer to last hidden layer
                output_vector3[j] = np.dot(self.weights_hidden_hidden[j], dropout_output_vector_hidden) 
                
                
                # calculates activation function of first hidden layer to last hidden layer
                output_vector_hidden_hidden = self.activation_function_class.whichActivationFunctionForwardPass(activation_function_string, output_vector3) # Choose between using activation function sigmoid or ReLu during the forward pass
                # calculate dropout of activation function of hidden layers –– i.e. pass output_vector_hidden_hidden as input to droput layer. IMPORTANT!
            
                self.dropout_class.setMode('train') 
                dropout_output_vector_hidden_hidden = self.dropout_class.dropoutForward(output_vector_hidden_hidden)
             '''


            
            
            
            #calculates weighted sum of last hidden layer to output layer
            output_vector2 = np.dot(self.weights_hidden_out, dropout_output_vector_hidden_hidden[len(self.weights_hidden_hidden)-1]) # THE PROBLEM IS dropout_output_vector_hidden NOT BEING OF APPRORPIATE SHAPE and DIMENSION
           
            #calculates activation function of last hidden layer to output layer –– IS THIS NEEDED??? CHECK TUTORIAL ANSWERS
            output_vector_network = self.activation_function_class.whichActivationFunctionForwardPass(activation_function_string, output_vector2) #Choose between using activation function sigmoid or ReLu during the forward pass
            
            # calculate dropout of activation function of last hidden layer to output layer –– i.e. pass output_vector_nestwork as input to droput layer. IMPORTANT! 
            # IS THIS NEEDED??? CHECK TUTORIAL ANSWERS
            # self.dropout_class.setMode('train') 
            # dropout_output_vector_hidden_out = self.dropout_class.dropoutForward(output_vector_network)
            
            
            #implement softmax as the last layer –– calculate the softmax of output layer after having aplied the other activation functions
            softmax_output_vector_network = self.softmax_layer_class.softmaxForward(self, output_vector_network)
        
        
        
        
        #BACKWARD PASS:
            # derivative of the activation functions of input to first hidden layer and of the output layer, which needs to be computed for later use in calculating hidden errors and weight updates 
            output_vector_network_derivate = self.activation_function_class.whichActivationFunctionBackwardPass(activation_function_string, output_vector_network) 
            dropout_output_vector_hidden_derivate = self.activation_function_class.whichActivationFunctionBackwardPass(activation_function_string, dropout_output_vector_hidden) 
            
            # implement softmax to calculate the gradient of the loss for the output layer of the network (result of forward pass).
            # This represents the difference between predicted softmax probabilities and the target labels, to be used to update the
            # weighted sum of last hidden layer to output layer (i.e. compute the weight updates of the result of forward pass???)
            # during the backward pass...,
            softmax_output_errors = self.softmax_layer_class.softmaxBackward(self, target_vector)
            
            # ...starting with that from the last hidden layer to the output layer, as the backward pass computes weight updates and 
            # calculates hidden errors iterating from the last layer to the input layer, across all hidden layers in reverse order (i.e. backpropagation). Thus, compute 
            # Weight updates of last hidden layer to output layer, according to tutorial 6.3: DONE
            tmp = softmax_output_errors * output_vector_network_derivate
            weight_update_hidden_output = self.learning_rate * np.dot(tmp, dropout_output_vector_hidden_hidden[len(self.weights_hidden_hidden)-1].T) 




            for j in range(1, len(self.weights_hidden_hidden)-1):
                # derivative of the activation function of first hidden layer to last hidden layer, which needs to be computed for later use in calculating hidden errors and weight updates 
                dropout_output_vector_hidden_hidden_derivate[j] = self.activation_function_class.whichActivationFunctionBackwardPass(activation_function_string, dropout_output_vector_hidden_hidden[j])
            # compute weight updates of last hidden layer to the first hidden layer. Although, tecnically, with this usage of j 
            # it would be from first hidden layer to the last hidden layer. Loop must iterate backwards to achieve "last hidden layer
            # to the first hidden layer" order. Which orderd is the correct one? For now, variable names may imply either order and 
            # must be corrected later if this don't match correct implementation once we find out which is the correct implementation.
            # Or, is the correct order already implemented by the .T transpose, cahnging original first hidden to last hidden to become 
            # last hidden to first hidden? Find out, do some testing. For the time being, use original first hidden to last hidden in 
            # case transpose already applies correct order for iteration in backpropagation.
            # I don't think we need to use softmax for anything other than last layer of network, whose weight updates have already been
            # calculated above as backpropagation requires reverse order. So, for the time being, comment out other useage of softmax and
            # make sure for anything other than last hidden layer to output layer, the weight updates are NOT calculated using softmax. 
            # I believe it goes like this, according to tutorial 6.3: Weight updates for last hidden layer to output layer (implemented above):
            # tmp = softmaxOfTheWholeNetwork (since softMax is only applied once) X whichActivationFunctionForwardPass (i.e. derivate of activation function, meaning backward whether sigmoid or relu) but where is dropout used !?
            # and then, this tmp is used in weight_update_hidden_output actual calculation/operation: 
            # weight_update_hidden_output = self.learning_rate * np.dot(tmp, dropout_output_vector_hidden_hidden[len(self.weights_hidden_hidden)-1].T), 
            # as weight update is done using the output from the previous layer (which in a hidden layer would be dropout, 
            # as opposed to directly using the activation function/output from the previous layer, as would be the case 
            # nowhere, I think, since we can't update the weights of output to soemthing else cause there is nothing else after output layer)
                tmp2hh[j] = dropout_output_vector_hidden_hidden[j] * dropout_output_vector_hidden_hidden_derivate[j]
                hidden_errors[j] = np.dot(self.weights_hidden_hidden.T, tmp2hh)
                tmp3hh[j] = hidden_errors[j] * dropout_output_vector_hidden_hidden_derivate[j]
                weight_update_hidden_hidden[j] = self.learning_rate * np.dot(tmp3hh, dropout_output_vector_hidden_hidden[j].T)
                
                # update the weights
                self.weights_hidden_hidden[j] += weight_update_hidden_hidden[j] #DONE

            # compute weight updates of input layer to first hidden layer -- copy from tutorial, not adapted to multiple layers, probably wrong.
            # weight_update_input_hidden = self.learning_rate * np.dot(softmax_output_errors, dropout_output_vector_hidden.T)




            # Weight updates for input layer to first hidden layer, according to tutorial 6.3: DONE
            # First, calculate hidden errors:
            # hidden_errors = np.dot(transpose of weights in next hidden layer 
            # (and because we are doing input to hidden, this "next" would be hidden to hidden, which is self.weights_hidden_hidden), tmp2) –– But, should this next hidden layer actually be input to hidden and thus, self.weights_in_hidden??? Double-check, but for the time being use self.weights_hidden_hidden.
            # where transpose of weights in next hidden layer = self.weights_hidden_hidden.T
            # and, where tmp2 = error from the next hidden layer (which previously, had been softmaxOfTheWholeNetwork, 
            # since in calculating "weight updates for last hidden layer to output layer" the "error from the next 
            # hidden layer" is the error from the output layer and the error of the output layer, instead of being 
            # dropout is softmax, but now, in calculating "weight updates for input layer to first hidden layer" the
            # "error from the next hidden layer" is the error from the first hidden layer, which is dropout instead 
            # of softmax –– dropout is applied to the hidden layers, softmax is applied to the output layer!!!) 
            # X derivative of the activation function used in the hidden layers 
            # (i.e. whichActivationFunctionBackwardPass of hidden (only of the first hidden layer or of the whole loop –– I don't know, double-check, but for now I will implement only for first hidden layer since it seems to be the most logic answer))
            # that is, tmp2 = dropout_output_vector_hidden X dropout_output_vector_hidden_derivate
            # Finally, hidden_errors = np.dot(self.weights_hidden_hidden.T, tmp2)
            tmp2 = dropout_output_vector_hidden * dropout_output_vector_hidden_derivate
            hidden_errors = np.dot(self.weights_hidden_hidden[1].T, tmp2)
            tmp3 = hidden_errors * dropout_output_vector_hidden_derivate
            # weight_update_input_hidden = self.learning_rate * np.dot(hidden_errors, dropout_output_vector_hidden #.T transpose )
            weight_update_input_hidden = self.learning_rate * np.dot(tmp3, input_vector.T)
                                                                   
            '''
            # calculate hidden errors:
            # hidden_errors = np.dot(transpose of weights in next hidden layer 
            # (and because we are doing input to hidden, this "next" would be hidden to hidden), tmp)
            # where tmp = same as tmp originally, which is softmaxOfTheWholeNetwork (since softMax is only applied once) X whichActivationFunctionBackwardPass (i.e. derivate of activation function, meaning backward whether sigmoid or relu)
            # (it is only later, after the hidden errors have been calculated, that tmp is overwriten)
            hidden_errors = np.dot(self.weights_hidden_out.T, softmax_output_errors)
            derivative_output = self.activation_function_class.whichActivationFunctionBackwardPass(activation_function_string, output_vector_hidden)  
            
            self.dropout_class.setMode(self, 'train')
            tmp = self.dropout_class.dropoutBackward(hidden_errors * derivative_output)
            wih_update = self.learning_rate * np.dot(tmp, input_vector.T)
            '''

            # update the weights:
            self.weights_in_hidden += weight_update_input_hidden # DONE
            # self.weights_hidden_hidden[j] += weight_update_hidden_hidden[j] # DONE, but inside loop
            self.weights_hidden_out += weight_update_hidden_output #DONE




    def run(self, input_vector, activation_function_string):
        # input_vector can be tuple, list or ndarray
        input_vector = np.array(input_vector, ndmin=2).T
        output_vector = [None]*len(input_vector)
        
        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = self.activation_function_class.whichActivationFunctionForwardPass(activation_function_string, output_vector)
        
        # may be wrong... if it fails try logic from forward pass 
        for i in range(0, len(self.weights_hidden_hidden)):
            output_vector = np.dot(self.weights_hidden_hidden[i], output_vector)
            output_vector = self.activation_function_class.whichActivationFunctionForwardPass(activation_function_string, output_vector)
        
        
        output_vector = np.dot(self.weights_hidden_out, output_vector)
        output_vector = self.activation_function_class.whichActivationFunctionForwardPass(activation_function_string, output_vector)
        
        return output_vector.T
    
    
    
    
    
    
    
    
    
    
    # Training in batches: 
        
        
    #TESTING:
'''
X, y = make_blobs(n_samples=60000, n_features=3072, centers=((1, 1,1), (5, 5,5)), cluster_std = 2)

#X = StandardScaler().fit_transform(X)
ax = plt.subplot(projection='3d')
ax.scatter3D( X[:,0], X[:,1], X[:,2], c=y)

#y = np.reshape(y,(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)
X_train = X_train.T

simple_network = NeuralNetwork(no_of_hidden_layers=5,
                               no_of_in_units=3, 
                               no_of_out_units=1, 
                               no_of_hidden_units=[4, 5, 6, 3, 2],
                               learning_rate=0.01,
                               dropoutRate=0.5,
                               activation_function_class=ActivationFunction,
                               softmax_layer_class = SoftmaxLayer,
                               dropout_class = Dropout)

simple_network.train(X_train, y_train, "sigmoidForward", epochs=100)

#y_hat = simple_network.run(X_test)

#y_hat[y_hat >0.5]=1
#y_hat[y_hat<0.5] =0
#print(sum(y_hat==y_test)/len(y_hat))


#ax = plt.subplot(projection='3d')
#ax.scatter3D( X_test[:,0], X_test[:,1], X_test[:,2], c=y_hat)

#plt.hist(y_hat)
plt.show()
'''
       
'''   
# Test from tutorial. Needs to be adapted.    
X, y = make_blobs( n_samples=5000, n_features=3, centers=((1, 1,1), (5, 5,5)), cluster_std = 2)

X = StandardScaler().fit_transform(X)
ax = plt.subplot(projection='3d')
ax.scatter3D( X[:,0], X[:,1], X[:,2], c=y)

y = np.reshape(y,(-1,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

simple_network = NeuralNetwork(no_of_hidden_layers=5,
                               no_of_in_units=3, 
                               no_of_out_units=1, 
                               no_of_hidden_units=[4, 5, 6, 3, 2],
                               learning_rate=0.01,
                               dropoutRate=0.5,
                               activation_function_class=ActivationFunction,
                               softmax_layer_class = SoftmaxLayer,
                               dropout_class = Dropout)

simple_network.train(X_train, y_train, "sigmoidForward", epochs=100)

#y_hat = simple_network.run(X_test)

#y_hat[y_hat >0.5]=1
#y_hat[y_hat<0.5] =0
#print(sum(y_hat==y_test)/len(y_hat))


#ax = plt.subplot(projection='3d')
#ax.scatter3D( X_test[:,0], X_test[:,1], X_test[:,2], c=y_hat)

#plt.hist(y_hat)
plt.show()
'''

