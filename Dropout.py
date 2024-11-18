import numpy as np

class Dropout:
    def __init__(self, p=0.5):
        self.p = p
        self.mask = None
        self.mode = 'train'
    
    def forward(self, x):
        "TODO"
    
    def backward(self, dout):
        "TODO"
    
    def setMode(self, mode):
        "TODO"