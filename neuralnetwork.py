import numpy as np

class NeuralNetwork:

    # constructor
    # get the number of layer sizes as a paramater and then
    # 1. initialize the weight matrix dimensions
    # 2. initialize the weight matrixes with values close to [-1, 1]
    # 3. initialize biases with zeros
    def __init__(self, layer_sizes):
        weight_dims = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]

        self.weights = [np.random.standard_normal(s) / s[1] ** .5 for s in weight_dims]
        self.biases = [np.zeros((s, 1)) for s in layer_sizes[1:]]

    # predict function
    # get the matrix a as a parameter
    # for each weight and bias, compute the activation function value
    # and store the node new values in a
    # return a
    def predict(self, a):
        for weight, bias in zip(self.weights, self.biases):
            z = np.matmul(weight, a) + bias
            a = self.activation(z)
        return a

    # activation function
    # use a sigmoid function 
    @staticmethod
    def activation(x):
        return 1 / (1 + np.exp(-x))