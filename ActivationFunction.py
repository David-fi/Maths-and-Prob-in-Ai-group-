import numpy as np 

def sigmoidForward(x):
    out = 1/ (1+ np.exp(-x)) #the sigmoid function
    cache = out 
    return out, cache

def simoidBackward(dout, cache):
    sig = cache
    dx = dout * sig * (1 - sig) #the derivative of the sigmoid function
    return dx

def reluForward(x):
    out = np.maximum(0,x)
    cache = x 
    return out, cache

def reluBackward(dout, cache):
    x = cache 
    dx = dout * (x > 0)
    return dx
