import numpy as np
import random


class CustomNeuralNetwork:

    # constructor
    # get the number of layer sizes as a paramater and then
    # 1. initialize the weight matrix dimensions
    # 2. set the number of layers of the network
    # 3. initialize the weight matrixes with values close to [-1, 1]
    # 4. initialize biases randomly

    def __init__(self, layer_sizes):
        weight_dims = [(a, b)
                       for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]
        print(weight_dims)
