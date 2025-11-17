"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning algorithm for a feedforward neural network. Gradients are calculated using backpropagation. Note that I have 
focused on keeping the code simple and curating detailed comments so one with a bit of background knowledge can understand the operations taking place. It is not optimized 
and omits many desirable features. 

"""

## import the python libraries that we need.
# NumPy is a powerful library for numerical computing in Python, particularly for operations involving arrays (lists) and matrices (spreadsheets). It provides support for 
# large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these data structures efficiently.
import numpy as np
# The random library provides functions for generating random numbers and performing random operations.
import random

# Define an activation function to introduce non-linearity to the output signals of our neurons so that our Network can differentiate between them to identify patterns
# in the data it can learn from. Here we use the sigmoid formula which contains the natural exponential base e (.exp). e is a number that models continuous change and thus 
# has a smooth curve allowing for differentiation, and a simple derivative that is easy to compute in a network of calculations. z is the variable name of the vector 
# used to hold the pre-activation value(s)-the linear combination of: inputs, weights, activations, and bias. e has an exponent of -z, which means that as z becomes more 
# positive, e to the -z becomes smaller, approaching zero but never reaching it. As z becomes more negative, e to the -z becomes larger, approaching infinity. Adding 1 to 
# the denominator ensures that it is always larger than the numerator which keeps the output of the sigmoid function between 0 and 1; interpretable as a probability. This
# means an output near .50 is uncertain, an output near 0 is very unlikely, and an output near 1 is very likely; resulting in a smooth S-shaped curve with the x axis as z 
# and the y axis as the output of the sigmoid function.
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

# Define a function to calculate the derivative of the sigmoid so that the network can learn from its mistakes. The calculation is as simple as multiplying the sigmoid 
# function by 1 minus itself. The result is a measure of how sensitive the output of the sigmoid function is to changes in its input (z). When the network outputs a 
# mistake, the direction and slope of it are measured, then carried back through the activation function of the output layer by the derivative of the sigmoid calculation. 
# From there a chain of interdependent calculations transmit error information backward through the rest of the network, providing each neuron with precise instructions 
# on how to adjust its parameters to reduce overall cost, thereby inducing "learning". This process is known as backpropagation. This sigmoid prime function is used every 
# time a layer in the network needs to pass the error signal backward through its sigmoid activation function. If we were to graph the derivative of the sigmoid function
# (aka sigmoid prime), we would see a bell-shaped curve. That curve peaks at 0.25 when z=0 and the sigmoid output is 0.5. It flattens as z approaches positive or negative 
# infinity. This means that the neuron learns the most when its output is uncertain and learns the least when its output is very certain.
def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


# In Python, "class" defines a new blueprint for creating "objects" (neural networks in our case, but it could be anything). Objects store data such as weights, biases and 
# other characteristics as attributes. They also hold functions which are called methods when they are defined inside of a class. Network is the name of our blueprint and 
# although it's descriptive, a class can be named anything. 'object' is required to be written as an argument in Python 2 but is optional in 3. 
class Network(object):
    # Whenever you create a new object, the __init__ method is automatically called to initialize its attributes. 'self' is an arbitrary but widely accepted naming 
    # convention used to refer to the specific object being called upon and is the first argument of any method in the class. 'sizes' is an ordinary argument name that the 
    # author chose and accepts a list of numbers as instructions for how to build the Neural Network.  
    def __init__(self , sizes):
        # Store the sizes list as an attribute of this network so when needed we can reference the number of its neurons per layer directly from the object rather than 
        # having to rely on an external variable.
        self.sizes = sizes
        # Store the length of the sizes list as an attribute so that we can use it to reference the number of layers in the network.
        self.num_layers = len(sizes)
        # To create the biases attribute we need to exclude the first number from the sizes list and iterate over the rest of it. The first number in sizes represents the 
        # input layer which holds the data we want our network to process, not biases. For each iteration after the input layer we need to generate that number of random 
        # values as biases and hold those biases in a vector. Once the loop has completed, we will have a list of bias vectors that our network can store as its attribute. 
        # sizes[1:] creates a new list that copies every entry from sizes after the first one. The for loop using y as the number of neurons per layer iterates over that 
        # list. NumPy accepts arguments y, 1 to create a column vector and uses its randn library to fill it with y random biases drawn from a standard normal distribution. 
        # The operaton occurs within list brackets to store the bias vectors.
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # To create the weights attribute we need to map the connections between every ordered pair of layers in the network. To do this we use an iterator called zip to 
        # form pairs based on the index positions of separate lists and then iterate over those indexed values with a for loop. The lists we are drawing from, sizes[:-1] 
        # and sizes[1:], slice the end and beginning off the sizes list respectively. This offsets the two lists so that the pair of values sharing the same index position 
        # contain the number of neurons in the previous layer, represented by x, and the number of neurons in the current layer, represented by y. We mark the connections 
        # between two layers with weights randomly drawn from a standard normal distribution using NumPy's randn function. For a given neuron the number of weights it has 
        # is equal to x. Therefore, our weights are organized into a matrix with y rows and x columns. The operation occurs within brackets to form the weight attribute.
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # Define a cost derivative method to calculate the direction (+/-) and slope of the "cost" also known as "loss", a performance metric used to reveal the distance 
    # between the network's output activations and its target values by finding the mean squared error between them. The cost derivative on the other hand tells us how 
    # sensitive the cost is to changes in the output activations and is calculated by simply subtracting the target values from the final layer's activation vector.
    def cost_derivative (self , output_activations , target_values):
        return (output_activations - target_values)

    # Define a method to compute how sensitive the cost is to changes in an individual weight or bias, known as that parameter's error gradient. The error gradient can be viewed 
    # in both parameter space (weights & biases) and activation space (pre-activations & output activations). Parameter space defines the shape of the network's cost function 
    # whereas activation space defines the flow of data through the network. Activations can be thought of as the "behavior" of the network, while the parameters are the 
    # "knobs" we can turn to tune that behavior. The number of dimensions that the cost surface exists within corresponds directly to the number of parameters in the network. 
    # In order to determine how each individual parameter can reduce the global cost we must calculate its local error gradient to discover which direction and how much to 
    # turn that knob. However, in practice, due to the high dimensional shape of the surface, a parameter may plateau or even increase its cost slightly on a given update.
    def compute_gradients (self, input_layer, target_values):
        # Create lists in the same shapes as the network’s biases and weights attributes but fill the arrays with zeros. The goal of this entire method is to replace the 
        # zeros with gradient values after measuring the cost of one training example; assuming that the network output an error. By creating blank copies we can operate 
        # directly on the actual attributes of the network yet store the resulting calculations in these variables. This effectively creates a one to one alignment between the
        # network's parameters and calculated gradients.
        example_b_grads = [np.zeros(b.shape) for b in self.biases]
        example_w_grads = [np.zeros(w.shape) for w in self.weights]
        ## This is the beginning of the forward pass. 
        # We start by initializing the placeholder for previous layer activations with the input layer since those encoded data are what we want our network to process and 
        # compute output activations for. 
        previous_layer_activations = input_layer
        # Create a list for the loop to store each layer's activation vector as we compute them and initialize it with the input layer.
        network_activations = [input_layer]
        # Store all the z vectors in a list to be used during back propagation.
        zs = []
        # This is the primary for loop that iterates through the network's attributes to compute the activations for each layer. Zip takes the index positions from the lists
        # of arrays containing the biases & weights attributes and iteratively combines them to form pairs of a bias and a weight array whose values can be transformed together 
        # in one line of code. b and w are representative of the values stored in the respective containers in the new pair.
        for b, w in zip(self.biases , self.weights):
            # Dot product takes the weight matrix and multiplies it with the previous layer's activation vector, then sums the product(s). The bias vector is then added to 
            # that result to find the pre-activation z value(s) for the layer. Inside of this operation, NumPy reuses the same activation vector for every row of the weight 
            # matrix, aligning each row’s weights with the activations from the same repeated vector so that the per-row dot products are computed efficiently without Python 
            # for-loops. Summing the per-neuron products across columns yields a single value per row, collapsing the 2 dimensional weight matrix into a 1 dimensional result 
            # vector. This new vector is then added to the bias vector to produce the pre-activation z vector for that layer.
            z = np.dot(w, previous_layer_activations)+b
            # Add the current layer's z vector to the list of z vectors.
            zs.append(z)
            # Apply the sigmoid function to the z vector to produce the activation vector for the layer.
            current_layer_activations = sigmoid (z)
            # Add the current layer's activation vector to the list of activation vectors for the network.
            network_activations.append(current_layer_activations)
            # Reset the previous layer activations variable to equal the current layer activation vector for the next iteration.
            previous_layer_activations = current_layer_activations
        ## This is the beginning of the backward pass.
        # Calculate the sensitivity of the cost to changes in the network's outputs, then measure the sensitivity of the outputs to changes in their pre-activation values.
        # Multiply these two sensitivities together to find the error at the output layer.
        error_signal = self.cost_derivative(network_activations [-1], target_values) * sigmoid_prime (zs [ -1])
        # The amount the bias needs to be adjusted to reduce cost is equal to the error signal it receives. Recall that it is added on after the dot product between the 
        # weights and the prior layer's activations when calculating z values. Thus the gradient of the bias is simply equal to the error signal.
        example_b_grads[-1] = error_signal
        # On the other hand, since each weight is multiplied by its corresponding activation from the previous layer during the forward pass, on the backward pass the weight 
        # gradients are found by multiplying the error(s) by the activation(s) of the previous layer. That is why activations[-2] is used, because starting from the end of 
        # the network (output layer) we need the activations from the one before it (the last hidden layer). However, if we look at our shapes: layer_error is a column vector
        # and so is activations[-2]. So we need to transpose activations[-2] to a row vector so that the dot product can multiply each error (row) by each activation (column)
        # which expands into a weight-gradient matrix.
        example_w_grads[-1] = np.dot(error_signal , network_activations[-2].transpose ())
        # xrange is an iterator used to generate a sequence of numbers. It accepts two arguments, the starting point which is 2 in this case and the number that you wish to 
        # iterate up to, but not include, which in our case is the number of layers. 
        for l in xrange (2, self.num_layers):
            # By placing a minus sign in front of l we can iterate in reverse order, so here the variable z starts out equal to the final hidden layer z container and iterates 
            # backwards to, but not including, the input layer.
            z = zs[-l]
            # Set a variable equal to the derivative of the sigmoid vector.
            sp = sigmoid_prime(z)
            # Since we are moving backward through the network we need to align the weights with the errors from the layer ahead,
            # transpose the weight matrix of each layer so that the rows (neurons) become columns, and the columns (weights) becomes rows. 
            # Now that every row is a weight and every column is a neuron when the dot product multiplies the transposed weight matrix by the layer_error vector,
            # it aligns each weight (row) with its corresponding error (column). This allows the dot product to sum the products of each weight*error for each neuron (row)
            # to produce a single error value for each neuron in the current layer (column). This error value is then multiplied by the derivative of the sigmoid function
            # (sp) to give the gradient of the cost function with respect to the pre-activation (z) values of the current layer. 
            # This gives us the layer_error for the current layer, which is then used to compute the gradients for the weights and biases of this layer.
            # This process is repeated for each layer moving backward through the network.
            error_signal = np.dot(self.weights[-l +1].transpose(), error_signal) * sp
            # Assign the layer_error to the appropriate position in the example_b_grads list.
            # -l indexes from the end of the list, so -2 is the second-to-last layer, -3 is the third-to-last, and so on.
            example_b_grads [-l] = error_signal
            # Transpose the activation vector from a column to a row so that the layer_error (column) can be multiplied by the activations (row)
            # to produce a weight-gradient matrix that matches the shape of the weight matrix between the current layer and the previous layer.
            # This is done for each layer moving backward through the network.
            example_w_grads [-l] = np.dot(error_signal , network_activations[-l -1].transpose())
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

    # create a method that sends an input vector 'a' through the network and returns the output vector.
    def feedforward(self , a):
        # Iterate through the hidden and output layers in the network and pair the weight matrix with the bias vector using zip.
        # This results in a tuple of (bias_vector, weight_matrix) for each layer. 
        for b, w in zip(self.biases , self.weights):
            # compute the activation function for the entire layer at once using numpy's dot product.
            a = sigmoid(np.dot(w, a)+b)
            return a

# [2,3,1] = 3 layers. The input layer has 2 neurons, the hidden layer has 3 neurons, and the output layer has 1 neuron.  
net = Network([2, 3, 1])