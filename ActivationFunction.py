import numpy as np

class ActivationFunction:
    def __init__(self, activationFunction):
        validActivation = ["sigmoid", "relu", "tanh"]
        if activationFunction not in validActivation:
            raise ValueError(f"{activationFunction} is not a valid activation function!")
        self.activationFunction = activationFunction

    def __reluForward(self, x):
        '''
        does the forward pass of the ReLU function
        x is the input arry 
        and then it outputs a tuple with the result of the  ReLU forward pass as well a cache used for backward pass
        cache this time is the input array 
        '''
        out = np.maximum(0, x)
        return out, x

    def __reluBackward(self, dout, cache):
        '''
        backward pass of the ReLU function
        x is just passing on the inpui arrat from forward pass using cache as the temporary store 
        dx is the gradient of the loss in respect to the input (being the array x)
        dout is the upstream gradient 
        '''
        # Ensure inputs are numpy arrays
        dout = np.array(dout)
        
        # Handle case where cache might be None
        if cache is None:
            raise ValueError("Cache cannot be None in reluBackward")
        
        x = np.array(cache)
        
        dx = dout * (x > 0) #derivative is 1 when x >0 otherwise it is 0
        return dx

    def __sigmoidForward(self, x):
        '''
        this function does the forward pass of the sigmoid function
        it takes an input array i just called x 
        and it returns a tuple with the result (out) of the sigmoid function
        as well as cache which we use in the backward pass, its just the same as the out 
        '''
        out = 1 / (1 + np.exp(-x)) # The sigmoid function
        cache = out
        return out, cache

    def __sigmoidBackward(self, dout, cache):
        '''
        does the backward pass of the sigmoid function
        i used d to show that its the derivative 
        so dx is the gradient of the loss with resepct to x (the input array)
        dout is the upstream gradient
        sig is just the sigmoid function hence why it equals cache
        '''
        # Ensure inputs are numpy arrays
        dout = np.array(dout)

        # Handle case where cache might be None
        if cache is None:
            raise ValueError("Cache cannot be None in sigmoidBackward")
        
        sig = np.array(cache)
        
        # Compute gradient
        dx = dout * sig * (1 - sig) # The derivative of the sigmoid function multiplied by the upstream gradient to get the proper flow of gradients
        return dx

    def __tanhForward(self, x):
        out = np.tanh(x) 
        cache = out 
        return out, cache

    def __tanhBackward(self, dout, cache):
        if cache is None:
            raise ValueError("Cache cannot be None in tanhBackward")
        tanh_val = np.array(cache)
        dx = dout * (1 - tanh_val ** 2)
        return dx

    def forward(self, x):
        if self.activationFunction == "relu":
            return self.__reluForward(x)
        elif self.activationFunction == "sigmoid":
            return self.__sigmoidForward(x)
        elif self.activationFunction == "tanh":
            return self.__tanhForward(x)

    def backward(self, dout, cache):
        if self.activationFunction == "relu":
            return self.__reluBackward(dout, cache)
        elif self.activationFunction == "sigmoid":
            return self.__sigmoidBackward(dout, cache)
        elif self.activationFunction == "tanh":
            return self.__tanhBackward(dout, cache)