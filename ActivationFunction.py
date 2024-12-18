import torch

class ActivationFunction:
    def __init__(self, activationFunction):
        validActivation = ["sigmoid", "relu", "tanh"]
        if activationFunction not in validActivation:
            raise ValueError(f"{activationFunction} is not a valid activation function!")
        self.activationFunction = activationFunction

    def forward(self, x):
        if self.activationFunction == "relu":
            return torch.relu(x), x
        elif self.activationFunction == "sigmoid":
            out = torch.sigmoid(x)
            return out, out
        elif self.activationFunction == "tanh":
            out = torch.tanh(x)
            return out, out

    def backward(self, dout, cache):
        if self.activationFunction == "relu":
            return dout * (cache > 0)
        elif self.activationFunction == "sigmoid":
            return dout * cache * (1 - cache)
        elif self.activationFunction == "tanh":
            return dout * (1 - cache**2)
