import numpy as np
import os
import matplotlib.pyplot as plt
import neuralnetwork as nn

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mnist.npz')
with np.load(path) as data: 
    training_images = data['training_images']
    training_labels = data['training_labels']

layer_sizes = (784, 5, 10)

network = nn.NeuralNetwork(layer_sizes)
network.print_accuracy(training_images, training_labels)