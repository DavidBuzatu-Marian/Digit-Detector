import numpy as np
import os
import matplotlib.pyplot as plt
import neuralnetwork as nn

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../dataset/mnist.npz')
with np.load(path) as data: 
    training_images = data['training_images']
    training_labels = data['training_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

layer_sizes = (784, 30, 10)


network = nn.NeuralNetwork(layer_sizes)
network.SGD(training_images, 30, 10, 3.0, test_data = test_images)


# network = nn.NeuralNetwork(layer_sizes)
# network.print_accuracy(training_images, training_labels)