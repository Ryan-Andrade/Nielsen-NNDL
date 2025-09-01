#import dependencies
import numpy as np
import random

# Create a python class called Network
class Network(object):
    # Define an initialization method to assign the attributes of the network (i.e. biases, weights,...)
    # 'self' refers to the Network
    # 'sizes' is a list argument. Each entry is the amount of neurons, and the # of entries=the # of layers.   
    def __init__(self , sizes):
        # create an attribute to express the number of layers in the network.
        self.num_layers = len(sizes)
        # create an attribute to capture the sizes list.
        self.sizes = sizes
        # the bias attribute uses numpy to create random values in a standard distribution with 0 mean and 1 standard deviation.
        # the y variable represents the number of neurons in the layer.
        # the 1 represents that each neuron has a single bias value.
        # this is done for each layer except the input layer (hence sizes[1:])
        # the result is a list of bias vectors
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # creates a weight matrix for each connection between layers:
        #   - x = number of neurons in the previous layer
        #   - y = number of neurons in the current layer
        # Each entry is drawn from a standard normal distribution (randn).
        # Using zip(sizes[:-1], sizes[1:]) pairs each (prev_layer_size, next_layer_size).
        # The result is a list of weight matrices.
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

# Create a network
net = Network([2, 3, 1])

# Test to see its weights
print(net.weights)

# Define a sigmoid function. z is a vector or matrix of weighted inputs
# depending on if you're updating your model one at a time or in batches.
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# create a function that sends an input 'a' through the network and returns the output.
def feedforward(self , a):
    # iterate through each bias and weight in the network, pair them together using zip.
    for b, w in zip(self.biases , self.weights):
        # set a equal to the activation function
        a = sigmoid(np.dot(w, a)+b)
        return a
    
# create a stochastic gradient descent function. 
# Training data is a list of tuples (x,y) representing the training inputs and the desired outputs.
# Test data is optional.
def SGD(self , training_data , epochs , mini_batch_size , learning_rate, test_data=None):
    if test_data:
        n_test = len(test_data)
        n = len(training_data)
    # for the argument (#) passed in for epochs, iterate through that many times.
    for j in xrange(epochs):
        random.shuffle(training_data)
        # set mini batches equal to slices of the training data based on the size set out in the argument.
        # k iterates from 0 to n (the length of the training data) in steps of mini_batch_size.
        mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            # calls the update_mini_batch function (defined next) to update the weights and biases
            #  of the network by the learning rate.
            self.update_mini_batch(mini_batch , learning_rate)
    if test_data:
        # if you decide to check the model's performance by including test data write a message to report 
        # how many test inputs the network got correct, (evaluate function defined later), after each training epoch. 
        print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
    else:
        # During training, print the epoch.
        print "Epoch {0} complete".format(j)

def update_mini_batch(self , mini_batch , learning_rate):
    # Create a vector full of zeros in the same shape as the biases in the network.
    bias_gradients = [np.zeros(b.shape) for b in self.biases]
    # Create a matrix full of zeros in the same shape as the weights in the network.
    weight_gradients = [np.zeros(w.shape) for w in self.weights]
    for input_data, target_value in mini_batch:
        # call the backprop function (defined later) and pass in the input data and target value.
        # for each training example, this returns the gradient for the cost function.
        b_gradient , w_gradient = self.backprop(input_data, target_value)
        bias_gradients = [nb+dnb for nb, dnb in zip(bias_gradients , b_gradient)]
        weight_gradients = [nw+dnw for nw, dnw in zip(weight_gradients , w_gradient)]
    self.weights = [w-(learning_rate/len(mini_batch))*nw for w, nw in zip(self.weights ,
        weight_gradients)]
    self.biases = [b-(learning_rate/len(mini_batch))*nb for b, nb in zip(self.biases , bias_gradients)]