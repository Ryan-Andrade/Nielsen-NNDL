"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on keeping the code simple and curating
detailed comments so one with a bit of background knowledge can understand the
operations taking place.  It is not optimized and omits many desirable features. 

"""

## import the python libraries that we need.
# Numpy is a powerful library for numerical computing in Python, particularly for operations involving arrays (lists) and matrices (spreadsheets). It provides 
# support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these data structures efficiently.
import numpy as np
# The random library provides functions for generating random numbers and performing random operations.
import random

# In Python, "class" defines a new blueprint for creating "objects" (neural networks in our case, but it could be anything).
# Objects hold data via attributes, such as weights and biases. They also hold functions which are called methods
# when they are defined inside of a class. 'object' is required to be passed as a parameter for Python 2 but is optional in 3.
# Network is the name of our class but it could be named anything.
class Network(object):
    # The __init__ method is automatically called to initialize the object's attributes whenever you create a new object of the class.
    # 'self' is an arbitrary name that refers to the specific object of the class and is the first parameter of 
    # any method in the class. It could be named anything, but 'self' is the widely accepted convention. 'sizes' is an ordinary parameter name 
    # that the author chose and accepts a list argument as instructions for how to build the Neural Network.  
    def __init__(self , sizes):
        # Each entry in sizes represents the amount of neurons per layer, thus the # of entries=the # of layers.
        self.num_layers = len(sizes)
        # Store the sizes list as an attribute of the network object.
        self.sizes = sizes
        # To create the biases attribute we need to skip the first the number in the sizes list because that is the input layer and it does not 
        # have any biases. Then we iterate through the rest of them and for each iteration we need to generate that number of random values as 
        # biases, hold those biases in a container, and then store that container as an entry into our attribute list. After that, we move onto the
        # next iteration and repeat this process until the list is complete. We skip the input layer with sizes[1:]. We iterate with a for loop
        # where the y variable assumes the number of neurons for the layer. Since there is only a single bias for each neuron, y is paired with 
        # a 1 to set the dimensions of the container. NumPy accepts these dimensions and uses its randn library to build the container with random
        # biases drawn from a standard normal distribution. This operation occurs within list brackets to store these bias containers. Most often 
        # these containers are vectors, but they can also be scalars.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # The container used to store a layer of weights can also be either a scalar or vector, but is most commonly a matrix. To create the weights 
        # attribute we need to model the connections between every ordered pair of layers in the network, so we use an iterator called zip in our for loop 
        # to form pairs based on the index positions of separate lists. The lists we are drawing from sizes[:-1] and sizes[1:] slice the end and beginning
        # off the sizes list respectively. This offsets the index positions of each list by one layer. The x and y variables assume the values of the 
        # indexed pair during every pass of the for loop. The order of the variables map to the order of the lists in zip, so x becomes the values in 
        # the list with the end sliced off and vice versa for y. Given that the values in the lists represent the number of neurons per layer, and since
        # the number of weights per neuron in the current layer is determined by the number of neurons in the previous layer we need to reverse the order
        # of the variables when passing them to NumPy's randn function. Doing so ensures that in a resulting container the # of rows, y, = the number of 
        # current layer neurons and the # of columns, x, = the number of previous layer neurons. The for loop then continues until zip has provided the 
        # values needed to create a weight container for every layer in the network. This operation occurs within list brackets storing the weight 
        # containters to form the weights attribute list.
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

# [2,3,1] = 3 layers. The input layer has 2 neurons, the hidden layer has 3 neurons, and the output layer has 1 neuron.  
net = Network([2, 3, 1])

# Define an activation function to introduce non-linearity to the output of our neurons so that our Network can differentiate between them to identify patterns
# in the data and learn from them. Here we use the sigmoid formula which contains the mathematical constant e '.exp' a number that has desirable properties such 
# as a smooth curve and a simple derivative to help our network learn. z is the pre-activation value, the linear combination of: inputs, weights, activations, and 
# bias. For a layer with a single neuron z is a scalar; for a layer of multiple neurons it's a vector. e has an exponent of -z, which means that as z becomes more 
# positive, e to the -z becomes smaller, approaching zero but never reaching it. As z becomes more negative, e to the -z becomes larger, approaching infinity. 
# Adding 1 to the denominator ensures that it is always larger than the numerator which keeps the output of the sigmoid function between 0 and 1; interpretable as a
# probability. This means an output near .50 is uncertain, an output near 0 is very unlikely, and an output near 1 is very likely resulting in a smooth S-shaped 
# curve with the x axis as z and the y axis as the output of the sigmoid function.
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# Define a function to calculate the derivative of the sigmoid to help the network learn from its mistakes. The calculation is as simple as multiplying the sigmoid 
# function by 1 minus itself because of e's unique mathematical properties. The result is a measure of how sensitive the output of the sigmoid function is to changes
# in its input (z). When the network outputs a mistake, the amount of error is measured and sent backwards through the network via a series of calculations in a process
# called backpropagation. The calculations act as instructions for the neurons on how to adjust their parameters closer to a value that has less error. The derivative 
# of the sigmoid calculation helps the neurons carry those instructions through the sigmoid transformation. If we were to graph the derivative of the sigmoid function,
# we would see a bell-shaped curve. That curve peaks at 0.25 when z=0 and the sigmoid output is 0.5. It flattens as z approaches positive or negative infinity. This 
# means that the neuron learns the most when its output is uncertain and learns the least when its output is very certain.
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

# The cost function derivative provides the starting error signal for backpropagation by subtracting the network's output activation(s) by the target value(s).
def cost_derivative (self , output_activations , target_values):
    return (output_activations - target_values)

# Define a function to compute the gradients of the cost function with respect to each weight and bias. These slopes indicate how each parameter should be adjusted to 
# reduce the cost and thereby improve the network's accuracy when used in training updates.
def compute_gradients (self, input_layer, target_values):
    # Create zero-filled containers with the same shapes as the network’s biases and weights attributes. These containers will be used to hold the gradient values for 
    # one network training example. Matching the shapes ensures each calculated gradient lines up one-to-one with its corresponding parameter so that as the function 
    # computes layer by layer, the gradients can be stored in containers to the correct positions. 

    # I want to edit this comment so that it takes the container operations perspective rather than the gradient level perspective
    example_b_grads = [np.zeros(b.shape) for b in self.biases]
    example_w_grads = [np.zeros(w.shape) for w in self.weights]
    ## This is the beginning of the forward pass. 
    # We choose the naming convention layer_activations because that is what is what we will be computing as we iterate through the network.
    # We set it equal to the input layer because the input layer itself does not have parameters (weights and biases) to compute an activation.
    # Activations are first computed starting from the hidden layer, so this just passes the raw input values forward as the starting point.
    # The input (x) and target (y) values that seed this computation are passed in from a chain of functions defined later.
    layer_activations = input_layer
    # Create a list for the loop to store each activation layer. Here it is clear that the input data
    # is what's received by the first layer, rather than actual activations, so the list does not begin empty.
    network_activations = [input_layer]
    # List to store all the z containers, layer by layer. Since no calculations have been done yet, it begins empty.
    zs = []
    # Zip takes the index position from the biases & weights attributes and iteratively forms pairs of a bias container and a weight container.
    # b and w assume the respective values held in each container of the new pair. Each pass of the for loop performs operations 
    # on the values held in these containers which together represent a layer of the network. 
    for b, w in zip(self.biases , self.weights):
        # Dot product takes each weight container and multiplies it with a layer-activation container, then sums the products
        # across columns for each row. The bias container is then added to that result to find the pre-activation z value(s)
        # for the layer. Looking inside the common containers (matrices and vectors): each row of a weight matrix corresponds
        # to one neuron in the layer, and each column position in that row corresponds to one input weight for that neuron.
        # These rows map directly to the layer’s biases and to the layer’s activations (both typically held as vectors).
        # During the multiplication step, NumPy reuses the same activation vector for every row of the weight matrix, aligning
        # each row’s columns with the vector’s entries so the per-row dot products are computed efficiently without Python for-loops.
        # Summing the per-row products across columns yields a single value per row, collapsing the (y, x) weight matrix with the
        # (x, 1) activation vector into a (y, 1) result vector. This new vector is then added to the bias vector to produce the
        # pre-activation z vector for that layer.
        z = np.dot(w, layer_activations)+b
        # Add the current layer's z vector to the list of z vectors.
        zs.append(z)
        # The sigmoid function is then applied to each entry in the z vector to produce the activation vector for the layer.
        layer_activations = sigmoid (z)
        # Add the current layer's activation vector to the list of activation vectors for the network.
        network_activations.append(layer_activations)
    ## This is the beginning of the backward pass.
    # Find the error at the output layer (activations[-1]) by computing the product of two terms:
    #   1. The derivative of the cost function (cost_derivative) which is simply the activation - the target value. This tells us what direction (+/-) and how much to
    #      adjust the output activations to reduce the cost.
    #   2. The derivative of the sigmoid function (sigmoid_prime) which measures how the output activations change as the pre-activation values (z) change.
    # The product of these two terms gives the gradient as a scalar for a single neuron output layer, or a vector for a multi-neuron output layer.
    layer_error = self.cost_derivative(network_activations [-1], target_values) * sigmoid_prime (zs [ -1])
    # Because the bias is simply added to the weighted inputs to find the pre-activation value, it can be adjusted directly by the error value therefore we can
    # assign the layer_error vector directly to the end of the example's biases gradients list, replacing the zeros that used to be there.
    example_b_grads[-1] = layer_error
    # On the other hand, since each weight is multiplied by its corresponding activation from the previous layer during the forward pass, on the backward pass the weight 
    # gradients are found by multiplying the error(s) by the activation(s) of the previous layer. That is why activations[-2] is used, because starting from the end of 
    # the network (output layer) we need the activations from the one before it (the last hidden layer). However, if we look at our shapes: layer_error is a column vector
    # and so is activations[-2]. So we need to transpose activations[-2] to a row vector so that the dot product can multiply each error (row) by each activation (column)
    # which expands into a weight-gradient matrix.
    example_w_grads[-1] = np.dot(layer_error , network_activations[-2].transpose ())
    # xrange is an iterator used to generate a sequence of numbers. It accepts two arguments, the starting point which is 2 in this case and the number that you wish to 
    # iterate up to, but not include, which in our case is the number of layers. 
    for l in xrange (2, self.num_layers):
        # By placing a minus sign in front of l we can iterate in reverse order, so here the variable z starts out equal to the final hidden layer z container and iterates 
        # backwards to, but not including, the input layer.
        z = zs[-l]
        # Set a variable equal to the derivative of the sigmoid function.
        sp = sigmoid_prime(z)
        # Since we are moving backward through the network we need to align the weights with the errors from the layer ahead,
        # transpose the weight matrix of each layer so that the rows (neurons) become columns, and the columns (weights) becomes rows. 
        # Now that every row is a weight and every column is a neuron when the dot product multiplies the transposed weight matrix by the layer_error vector,
        # it aligns each weight (row) with its corresponding error (column). This allows the dot product to sum the products of each weight*error for each neuron (row)
        # to produce a single error value for each neuron in the current layer (column). This error value is then multiplied by the derivative of the sigmoid function
        # (sp) to give the gradient of the cost function with respect to the pre-activation (z) values of the current layer. 
        # This gives us the layer_error for the current layer, which is then used to compute the gradients for the weights and biases of this layer.
        # This process is repeated for each layer moving backward through the network.
        layer_error = np.dot(self.weights[-l +1].transpose(), layer_error) * sp
        # Assign the layer_error to the appropriate position in the example_b_grads list.
        # -l indexes from the end of the list, so -2 is the second-to-last layer, -3 is the third-to-last, and so on.
        example_b_grads [-l] = layer_error
        # Transpose the activation vector from a column to a row so that the layer_error (column) can be multiplied by the activations (row)
        # to produce a weight-gradient matrix that matches the shape of the weight matrix between the current layer and the previous layer.
        # This is done for each layer moving backward through the network.
        example_w_grads [-l] = np.dot(layer_error , network_activations[-l -1].transpose())
    return ( example_b_grads , example_w_grads )

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
# Training data is a list of tuples (x, y) representing the input layer as a vector (x) 
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
        print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
    else:
        # During training, print the epoch.
        print ("Epoch {0} complete".format(j))

# create a function that sends an input vector 'a' through the network and returns the output vector.
def feedforward(self , a):
    # Iterate through the hidden and output layers in the network and pair the weight matrix with the bias vector using zip.
    # This results in a tuple of (bias_vector, weight_matrix) for each layer. 
    for b, w in zip(self.biases , self.weights):
        # compute the activation function for the entire layer at once using numpy's dot product.
        a = sigmoid(np.dot(w, a)+b)
        return a

