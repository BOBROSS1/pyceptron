# pyceptron
import numpy as np


def sigmoid(x):
    ''' normalizes output between 0 and 1 '''
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    ''' takes derivative of output which produces the 'amount' by which
        the weight should be adjusted (neurons with high certainty should
        bet adjusted less) '''
    return sigmoid(x) * (1 - sigmoid(x))


# sample training data
training_inputs = np.array([[0, 0, 1],
                            [1, 1, 1],
                            [1, 0, 1],
                            [0, 1, 1]])

# sample training outputs (truth)
training_outputs = np.array([[0, 1, 1, 0]]).T

# seed for random nr generator
np.random.seed(1)

# random weights produced as vector (3*1)
synaptic_weights = 2 * np.random.random((3, 1)) - 1

print('Random starting synaptic weights: ')
print(synaptic_weights)


for iteration in range(1):
    # training data goes in
    input_layer = training_inputs

    # output is the dot product of the training data with the
    # random initialized weights vector per row of the trainig data.
    # out comes a vector of outputs for each row of the input (normalized
    # by the sigmoid function).
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))

    # error between truth and guess is calculated(truth - output)
    # error is a 4*1 vector
    error = training_outputs - outputs

    # the adjusment needed is calculated by multiplying the error
    # with the amount the neuron should be changed. This amount is
    # calculated by the derivative of the output (of that neuron). If the
    # output was large, the weight was heavy meaning the neuron was quite
    # confident. Confident neurons should be adjusted less. Taking confidence
    # into account is done by by multipying the error with the derivative
    # since the derivative of a large number (in the sigmoid graph) results in a
    # smaller number. Adjustments is a 4*1 matrix.
    adjustments = error * sigmoid_derivative(outputs)

    # update the synaptic weights by multiplying them with the adjustments
    synaptic_weights += np.dot(input_layer.T, adjustments)

print("Synaptic weights after training: ")
print(synaptic_weights)

print("These trained weights produce outputs: ")
print(outputs)
