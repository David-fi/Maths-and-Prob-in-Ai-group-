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
        
        for i in range(len(no_of_hidden_units) - 1):
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
        
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T
      #  Dropout.training = True
    
        for i in range(epochs): # loop allows to perform the following caclulation for every layer of the NN
        # i.e. performs a forward pass where the weighted sums and activation functions (sigmoidForward or ReLu) are
        # calculated and a backward pass where weighted updates are calculated using the derivative of the activation
        # function (wether that is sigmoidBackward or reluBackward)
        
        #FORWARD PASS:
            #calculates weighted sum of hidden layer
            output_vector1 = np.dot(self.weights_in_hidden, input_vector)
            # calculates activation function of hidden layer
            output_vector_hidden = self.activation_function_class.whichActivationFunctionForwardPass(activation_function_string, output_vector1) # Choose between using activation function sigmoid or ReLu during the forward pass
            # calculate dropout of activation function of hidden layer –– i.e. pass output_vector_hidden as input to droput layer. IMPORTANT!
            
            self.dropout_class.setMode('train') 
            
            dropout_output_vector_hidden = self.dropout_class.dropoutForward(output_vector_hidden)
            
            
            #calculates weighted sum of output layer
            output_vector2 = np.dot(self.weights_hidden_out, dropout_output_vector_hidden)
            #calculates activation function of output layer –– IS THIS NEEDED??? CHECK TUTORIAL ANSWERS
            output_vector_network = self.activation_function_class.whichActivationFunctionForwardPass(activation_function_string, output_vector2) #Choose between using activation function sigmoid or ReLu during the forward pass
            
            #implement softmax as the last layer –– calculate the softmax of output layer after having aplied the other activation functions
            softmax_output_vector_network = self.softmax_layer_class.softmaxForward(self, output_vector_network)
        
        
        #BACKWARD PASS:
            #implement softmax as the last layer –– calculate the softmax of output layer after having aplied the other activation functions
            softmax_output_errors = self.softmax_layer_class.softmaxBackward(self, target_vector)
    
            who_update = self.learning_rate * np.dot(softmax_output_errors, dropout_output_vector_hidden.T)

            # calculate hidden errors:
            hidden_errors = np.dot(self.weights_hidden_out.T, softmax_output_errors)

            derivative_output = self.activation_function_class.whichActivationFunctionBackwardPass(activation_function_string, output_vector_hidden)  
            
            tmp = self.dropout_class.setMode(self, 'train').dropoutBackward(hidden_errors * derivative_output)
            wih_update = self.learning_rate * np.dot(tmp, input_vector.T)

            # update the weights:
            self.weights_in_hidden += wih_update
            self.weights_hidden_out += who_update


    def run(self, input_vector, activation_function_string):
        # input_vector can be tuple, list or ndarray
        input_vector = np.array(input_vector, ndmin=2).T
        
        output_vector = np.dot(self.weights_in_hidden, input_vector)
        output_vector = self.activation_function_class.whichActivationFunctionForwardPass(activation_function_string, output_vector)
        
        '''
        for i in range(len(no_of_hidden_units) - 1):
        output_vector = np.dot(self.weights_hidden_hidden, input_vector)
        output_vector = self.activation_function_class.whichActivationFunctionForwardPass(activation_function_string, output_vector)
        '''
        
        output_vector = np.dot(self.weights_hidden_out, output_vector)
        output_vector = self.activation_function_class.whichActivationFunctionForwardPass(activation_function_string, output_vector)
        
        return output_vector.T
    
    
    # Training in batches: 
        
        
    #TESTING:
        
        
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
                               activation_function_class=ActivationFunction)
simple_network.train(X_train,y_train, "sigmoidForward")

y_hat = simple_network.run(X_test)

y_hat[y_hat >0.5]=1
y_hat[y_hat<0.5] =0
print(sum(y_hat==y_test)/len(y_hat))


ax = plt.subplot(projection='3d')
ax.scatter3D( X_test[:,0], X_test[:,1], X_test[:,2], c=y_hat)

plt.hist(y_hat)
plt.show()
    
    
