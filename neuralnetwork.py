import numpy as np

class NeuralNetwork:

    # constructor
    # get the number of layer sizes as a paramater and then
    # 1. initialize the weight matrix dimensions
    # 2. set the number of layers of the network
    # 3. initialize the weight matrixes with values close to [-1, 1]
    # 4. initialize biases randomly
    def __init__(self, layer_sizes):
        weight_dims = [(a, b) for a, b in zip(layer_sizes[1:], layer_sizes[:-1])]

        self.nr_layers = len(layer_sizes)

        # self.weights = [np.random.standard_normal(s) / s[1] ** .5 for s in weight_dims]
        self.weights = [np.random.randn(s[0], s[1]) for s in weight_dims]
        # self.biases = [np.zeros((s, 1)) for s in layer_sizes[1:]]
        self.biases = [np.random.randn(s, 1) for s in layer_sizes[1:]]
        print(self.weights[0])

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

    # SGD method
    # it uses stochastic gradient descent to learn the parameters
    # it uses code from http://neuralnetworksanddeeplearning.com/chap1.html
    # it gets a list of tuples (training_data) and it can show a progress if test_data is provided
    def SGD(self, training_data, epochs, mini_batch_size, learning_rate,
            test_data = None):
            # check if test data exists and take its length
            if (test_data):
                nr_test = len(test_data)
            size_of_training = len(training_data)
            for j in range(epochs):
                random.shuffle(training_data)
                mini_batches = [
                    training_data[k:k + mini_batch_size for k in range(0, size_of_training, mini_batch_size)]
                ]
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, learning_rate)
                if test_data:
                    print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), nr_test)
                else:
                    print "Epoch {0} complete".format(j)


    def print_accuracy(self, images, labels):
        predictions = self.predict(images)
        num_correct = sum([np.argmax(a) == np.argmax(b)  for a,b in zip(predictions, labels)])
        print('{0}/{1} accuracy: {2}%'.format(num_correct, len(images), (num_correct / len(images) * 100)) )

    # activation function
    # use a sigmoid function 
    @staticmethod
    def activation(x):
        return 1 / (1 + np.exp(-x))