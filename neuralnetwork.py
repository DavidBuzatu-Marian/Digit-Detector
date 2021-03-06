import numpy as np
import random


class NeuralNetwork:

    # constructor
    # get the number of layer sizes as a paramater and then
    # 1. initialize the weight matrix dimensions
    # 2. set the number of layers of the network
    # 3. initialize the weight matrixes with values close to [-1, 1]
    # 4. initialize biases randomly
    def __init__(self, layer_sizes):
        weight_dims = [(a, b)
                       for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]

        self.nr_layers = len(layer_sizes)

        # self.weights = [np.random.standard_normal(s) / s[1] ** .5 for s in weight_dims]
        self.weights = [np.random.randn(s[0], s[1]) for s in weight_dims]
        # self.biases = [np.zeros((s, 1)) for s in layer_sizes[1:]]
        self.biases = [np.random.randn(s, 1) for s in layer_sizes[1:]]

    # predict function
    # get the matrix a as a parameter
    # for each weight and bias, compute the activation function value
    # and store the node new values in a
    # return a
    def predict(self, a):
        for weight, bias in zip(self.weights, self.biases):
            z = np.matmul(weight, a) + bias
            a = self.sigmoid(z)
        return a

    # SGD method
    # it uses stochastic gradient descent to learn the parameters
    # it uses code from http://neuralnetworksanddeeplearning.com/chap1.html
    # it gets a list of tuples (training_data) and it can show a progress if test_data is provided
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate,
            test_data=None):
        # check if test data exists and take its length
        if (test_data.any()):
            nr_test = len(test_data)
        size_of_training = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, size_of_training, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j,
                                                    self.evaluate(test_data), nr_test))
            else:
                print("Epoch {0} complete".format(j))

    # update_mini_batch
    # method used to run a single iteration of gradient descent on the provided batch

    def update_mini_batch(self, mini_batch, learning_rate):
        len_mini_batch = len(mini_batch)

        # create new matrices for the new layer of neurons
        new_w = [np.zeros(weight.shape) for weight in self.weights]
        new_b = [np.zeros(bias.shape) for bias in self.biases]
        print(mini_batch)
        for X, y in mini_batch:
            # get the updated weight and biases
            delta_new_w, delta_new_b = self.backprop(X, y)
            # update weights and biases with the results of backprop
            new_w = [nw + d_nw for nw, d_nw in zip(new_w, delta_new_w)]
            new_b = [nb + d_nb for nb, d_nb in zip(new_b, delta_new_b)]
        # update weights and biases based on result of backprop on mini batch
        self.weights = [w - (learning_rate / len_mini_batch)
                        * nw for w, nw in zip(self.weights, new_w)]
        self.biases = [b - (learning_rate / len_mini_batch) *
                       nb for b, nb in zip(self.biases, new_b)]

    def print_accuracy(self, images, labels):
        predictions = self.predict(images)
        num_correct = sum([np.argmax(a) == np.argmax(b)
                           for a, b in zip(predictions, labels)])
        print('{0}/{1} accuracy: {2}%'.format(num_correct,
                                              len(images), (num_correct / len(images) * 100)))

    # backprop function
    # steps:
    # 1. calculate the activation of all layers (forward prop.)
    # 2. compute delta of last layer (delta[L])
    # 3. compute intermediary delta (delta[L - 1], delta[L - 2], etc.)
    # 4. compute Deltas using delta and activation of layers
    # 5. update weight and bias
    def backprop(self, X, y):
        # init weights and biases to zero
        new_w = [np.zeros(w.shape) for w in self.weights]
        new_b = [np.zeros(b.shape) for b in self.biases]

        # feedforward part
        activation = X  # initialize to X
        # an array because we have N layers of activations
        activation_results = [X]
        z_results = []  # an array for all z's in layers

        for weight, bias in zip(self.weights, self.biases):
            z = np.dot(weight, activation) + bias
            z_results.append(z)  # append z to list of z's
            activation = sigmoid(z)  # compute activation
            # append to list of activations
            activation_results.append(activation)

        # backward part
        delta = (activation_results[-1] - y) * sigmoid_d(z_results[-1])
        new_b[-1] = delta  # last layer of bias becomes delta
        # compute the last layer of weights
        new_w[-1] = compute_weights(delta, activation_results[-2].transpose())

        for L in range(2, self.nr_layers):
            # compute delta and update weight and bias for each layer
            z = z_results[-L]
            z_d = sigmoid_d(z)
            delta = np.dot(self.weights[-L + 1].transpose(), delta) * z_d
            new_b[-L] = delta
            new_w[-L] = compute_weights(delta,
                                        activation_results[-L - 1].transpose())
        return (new_w, new_b)

    # evaluation function
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(X)), y)
                        for (X, y) in test_data]
        return sum(int(X == y) for (X, y) in test_results)

    def compute_weights(self, delta, activation_results):
        return np.dot(delta, activation_results)

    # activation function
    # use a sigmoid function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # derivative of sigmoid
    def sigmoid_d(self, z):
        return sigmoid(z) * (1 - sigmoid(z))
