import numpy as np


input_set = np.array([[0,1,0],
                      [0,0,1],
                      [1,0,0],
                      [1,1,0],
                      [1,1,1],
                      [0,1,1],
                      [0,1,0]])

labels = np.array([[1, 0, 0, 1, 1, 0, 1]])

labels = labels.reshape(7, 1)

np.random.seed(42)
weights = np.random.rand(3, 1)
bias = np.random.rand(1)
lr = 0.05


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_d(x):
    return sigmoid(x) * (1 - sigmoid(x))

EPOCH = 25000

for epoch in range(EPOCH):
    inputs = input_set
    cost = np.dot(inputs, weights) + bias
    z = sigmoid(cost)
    error = z - labels
    print(error.sum())

    d_cost = error
    d_pred = sigmoid_d(z)
    z_d = d_cost * d_pred
    inputs = input_set.T
    weights = weights - lr*np.dot(inputs, z_d)

    for num in z_d:
        bias -= (lr * num)

single_pt = np.array([0, 1, 0])
result = sigmoid(np.dot(single_pt, weights) + bias)
print(result)
