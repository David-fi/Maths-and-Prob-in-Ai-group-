import numpy as np 

class ActivationFunction:
    
    def __init__(self, seed=None):
        self.seed = seed
        
    @staticmethod
    def sigmoidForward(x):
         '''
         this function does the forward pass of the sigmoid function
         it takes an input array i just called x 
         and it returns a tuple with the result (out) of the sigmoid function
         as well as cache which we use in the backward pass, its just the same as the out 
         '''
         out = 1 / (1 + np.exp(-x)) #the sigmoid function
         cache = out 
         return out, cache
        
    @staticmethod
    def sigmoidBackward(dout, cache):
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
        dx = dout * sig * (1 - sig) #the derivative of the sigmoid function multiplied by the upstream gradient to get the proper flow of gradients
        return dx
        
    @staticmethod   
    def reluForward(x):
        '''
        does the forward pass of the ReLU function
        x is the input arry 
        and then it outputs a tuple with the result of the  ReLU forward pass as well a cache used for backward pass
        cache this time is the input array 
        '''
        out = np.maximum(0,x)
        cache = x 
        return out, cache
    
    @staticmethod
    def reluBackward(dout, cache):
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

    @staticmethod
    def whichActivationFunctionForwardPass(activation_function_string, output_vector):
        if activation_function_string == "sigmoidForward":
            return ActivationFunction.sigmoidForward(output_vector)
        
        elif activation_function_string == "reluForward":
            return ActivationFunction.reluForward(output_vector)

    @staticmethod
    def whichActivationFunctionBackwardPass(activation_function_string, output_vector, cache=None):
        if activation_function_string == "sigmoidBackward":
            if cache is None:
                raise ValueError("Cache is None for sigmoidBackward")
            return ActivationFunction.sigmoidBackward(output_vector, cache)
        elif activation_function_string == "reluBackward":
            if cache is None:
                raise ValueError("Cache is None for reluBackward")
            return ActivationFunction.reluBackward(output_vector, cache)
        else:
            raise ValueError(f"Unsupported activation function: {activation_function_string}")
