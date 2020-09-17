import numpy as np
import mnist_loader
import matplotlib.pyplot as plt
import neuralnetwork as nn

traning_data, validation_data, test_data = mnist_loader.load_data_wrapper()

nework = nn.NeuralNetwork([784, 30, 10])
network.SGD(traning_data, 30, 10, 3.0, test_data = test_data)
