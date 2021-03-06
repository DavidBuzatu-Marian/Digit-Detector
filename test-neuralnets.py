import numpy as np
import neuralnetwork as nn


layer_sizes = (5, 3, 6)
x = np.ones((layer_sizes[0], 1))

network = nn.NeuralNetwork(layer_sizes)
prediction = network.predict(x)

# print(prediction)