"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

# import dependencies
import numpy as np
import random

# Create a python class called Network
class Network(object):
    ## Define an initialization method to assign the attributes of the network (i.e. biases, weights,...)
    # 'self' refers to the Network
    # 'sizes' is a list argument. Each entry is the amount of neurons, and the # of entries=the # of layers. 
    # For example, [2,3,1] = 3 layers. The input layer has 2 neurons, the hidden layer has 3 neurons, and the output layer has 1 neuron.  
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
        # create a weight matrix for each connection between layers:
        #   - x = number of neurons in the previous layer
        #   - y = number of neurons in the current layer
        # Each entry is drawn from a standard normal distribution (randn).
        # Using zip(sizes[:-1], sizes[1:]) pairs each (prev_layer_size, next_layer_size).
        # The result is a list of weight matrices.
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

# Create a network
net = Network([2, 3, 1])

# Test to see its weights & biases
print(net.weights)
print(net.biases)

# Define a sigmoid function that introduces non-linearity to the network.
# z is the pre-activation value: the linear combination of: inputs, weights, activations, and bias. 
# For a single neuron z is a scalar; for a layer of neurons it's a NumPy array. 
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# Compute the gradients of the cost function with respect to each weight and bias. 
# These gradients indicate how each parameter should be adjusted to reduce the cost 
# (and thereby improve the network's accuracy when used in training updates).
def compute_gradients (self , input_layer, target_value):
    batch_total_b_grads = [np.zeros (b.shape ) for b in self.biases]
    batch_total_w_grads = [np.zeros (w.shape ) for w in self.weights]
    # This is the beginning of the forward pass. The input layer, is given to the for loop to calculate the activations for
    # the first hidden layer. Hence, why we set activation equal to the input layer.
    activation = input_layer
    # List to store activations, layer by layer. Here it is clear that
    # input_data is what is received by the first layer, rather than actual activations.
    activations = [input_layer]
    # List to store all the z vectors, layer by layer.
    zs = []
    
    for b, w in zip( self.biases , self.weights ):
        # the dot product accepts a matrix of weights and multiplies each row in the matrix
        #  by the activation vector, then it adds the products (columns) of each row together.
        # the bias vector is then added to each neuron in the layer.
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

# Model training will employ sequential updates to the weights and biases using mini-batches of training data.
# This function will update the network's weights and biases.
def update_mini_batch(self , mini_batch , learning_rate):
    # Create a list full of vectors containing zero values in the same shape as 
    # the biases in the network, (bias => vector => layer => list). This list will hold
    # the running total of the bias gradients for the mini-batch.
    batch_total_b_grads = [np.zeros(b.shape) for b in self.biases]
    # Create a list full of matrices containing zero values in the same shape as 
    # the weights in the network, (inputs => weights => neuron => matrix => layer => list).
    # This list will hold the running total of the weight gradients for the mini-batch.
    batch_total_w_grads = [np.zeros(w.shape) for w in self.weights]
    for input_layer, target_value in mini_batch:
        # For each training example, pass in the input layer and target value to the compute gradients
        # function (defined next). This returns a list of the gradient slope for every 
        # variable in the network in the same shape as its respective data structure.
        example_b_gradients , example_w_gradients = self.compute_gradients(input_layer, target_value)
        # Once you receive the gradients list for this example, match it to the list containing the sum 
        # of the mini-batch gradients using zip, and add them to the running total.
        batch_total_b_grads = [example_b_grads+batch_sum_b_grads for example_b_grads, batch_sum_b_grads in zip(batch_total_b_grads , example_b_gradients)]
        batch_total_w_grads = [example_w_grads+batch_sum_w_grads for example_w_grads, batch_sum_w_grads in zip(batch_total_w_grads , example_w_gradients)]
    # weight matrix = connection between layers. update applies to entire matrix.
    self.weights = [weight_matrix-(learning_rate/len(mini_batch))*weight_gradient_matrix for weight_matrix, weight_gradient_matrix in zip(self.weights ,
        batch_total_w_grads)]
    # bias vector = entire layer
    self.biases = [bias_vector-(learning_rate/len(mini_batch))*bias_gradient_vector for bias_vector, bias_gradient_vector in zip(self.biases , batch_total_b_grads)]

def evaluate(self, test_data):
    """Return the number of test inputs for which the neural
    network outputs the correct result. Note that the neural
    network's output is assumed to be the index of whichever
    neuron in the final layer has the highest activation."""
    test_results = [(np.argmax(self.feedforward(x)), y)
                    for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)

# This is the core method to train the neural network using mini-batch stochastic gradient descent. 
# It accepts the tunable hyperparameters and calls the other functions we defined above. 
# Training data is a list of tuples (x, y) representing the training inputs (x) 
# and the target values (y). Test data is optional.
def SGD(self , training_data , epochs , mini_batch_size , learning_rate, test_data=None):
    if test_data:
        n_test = len(test_data)
        n = len(training_data)
    # epochs is passed as an integer. xrange converts the integer into an iterable object. 
    # j then takes on each integer (0 to epochs-1) to track which epoch the network is on, 
    # and is later used in print statements to report training progress.
    for j in xrange(epochs):
        # At the start of each epoch, shuffle the entire training set so that mini-batches 
        # are formed from different groupings each time. This prevents the network from 
        # picking up false patterns from the fixed order of the data.
        random.shuffle(training_data)
        # set mini_batches equal to a list comprised of slices of the training data 
        # based on the size set out in the argument. 
        # k iterates from 0 to n (the length of the training data) in steps of mini_batch_size.
        mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
        for mini_batch in mini_batches:
            # call the update_mini_batch function to update the weights and biases
            # of the network by the learning rate.
            self.update_mini_batch(mini_batch , learning_rate)
    if test_data:
        # Including test data calls the evaluate function to create a running tally of model performance per epoch.
        # This drastically slows down training, so only include it if you need to see progress. 
        print "Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test)
    else:
        # During training, print the epoch.
        print "Epoch {0} complete".format(j)

# create a function that sends an input vector 'a' through the network and returns the output vector.
def feedforward(self , a):
    # Iterate through the hidden and output layers in the network and pair the weight matrix with the bias vector using zip.
    # This results in a tuple of (bias_vector, weight_matrix) for each layer. 
    for b, w in zip(self.biases , self.weights):
        # compute the activation function for the entire layer at once using numpy's dot product.
        a = sigmoid(np.dot(w, a)+b)
        return a

