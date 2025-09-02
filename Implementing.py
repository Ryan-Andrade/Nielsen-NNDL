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
    # Create a list full of vectors containing zero values in the same shape as 
    # the biases in the network, (bias => vector => layer => list).
    batch_total_b_grads = [np.zeros(b.shape) for b in self.biases]
    # Create a list full of matrices containing zero values in the same shape as 
    # the weights in the network, (inputs => weights => neuron => matrix => layer => list).
    batch_total_w_grads = [np.zeros(w.shape) for w in self.weights]
    for input_data, target_value in mini_batch:
        # call the backprop function (defined later) and pass in the input data and target value.
        # for each training example, this returns the gradient slope for every variable in the network
        # in the same shape as its respective list defined above.
        example_b_gradients , example_w_gradients = self.backprop(input_data, target_value)
        # once you receive this example's gradients, match them with the list and add each gradient to
        # its area of the list (network proxy) to create a running total of the gradients for the mini-batch.
        batch_total_b_grads = [example_b_grads+batch_sum_b_grads for example_b_grads, batch_sum_b_grads in zip(batch_total_b_grads , example_b_gradients)]
        batch_total_w_grads = [example_w_grads+batch_sum_w_grads for example_w_grads, batch_sum_w_grads in zip(batch_total_w_grads , example_w_gradients)]
    # subtract each variable in the network by the learning rate, divided by the length of the mini-batch,
    # multiplied by the corresponding variable's gradient slope in the batch total list. 
    # Then reassign the network's variable to this new value.
    self.weights = [w-(learning_rate/len(mini_batch))*weight_gradient_slope for w, weight_gradient_slope in zip(self.weights ,
        batch_total_w_grads)]
    self.biases = [b-(learning_rate/len(mini_batch))*bias_gradient_slope for b, bias_gradient_slope in zip(self.biases , batch_total_b_grads)]
# TODO: replace spaces with tabs for proper indentation
def backprop (self , input_data, target_value):
    batch_total_b_grads = [np. zeros (b. shape ) for b in self . biases ]
    batch_total_w_grads = [np. zeros (w. shape ) for w in self . weights ]
    # feedforward
    activation = input_data
    activations = [input_data] # list to store all the activations , layer by layer
    zs = [] # list to store all the z vectors , layer by layer
    for b, w in zip( self .biases , self . weights ):
        z=np. dot (w, activation )+b
        zs. append (z)
        activation = sigmoid (z)
        activations . append ( activation )
    # backward pass
    delta = self . cost_derivative ( activations [-1], target_value) * sigmoid_prime (zs [ -1])
    batch_total_b_grads [ -1] = delta
    batch_total_w_grads [ -1] = np.dot(delta , activations [ -2]. transpose ())
    for l in xrange (2, self . num_layers ):
        z = zs[-l]
        sp = sigmoid_prime (z)
        delta = np.dot( self . weights [-l +1]. transpose () , delta ) * sp
        batch_total_b_grads [-l] = delta
        batch_total_w_grads [-l] = np.dot(delta , activations [-l -1]. transpose ())
    return ( batch_total_b_grads , batch_total_w_grads )