import random
from numpy import *
from functools import reduce


def sigmoid(x):
    """
    Computes the sigmoid activation function.

    The sigmoid function maps any real-valued number into the range (0, 1),
    which is commonly used as an activation function in neural networks.

    Args:
        x (float): The input value.

    Returns:
        float: The result of applying the sigmoid function to the input.
    """
    return 1.0 / (1 + exp(-x))


class Node(object):
    """
    Represents a node (neuron) in a neural network layer.

    Attributes:
        layer_index (int): The index of the layer this node belongs to.
        node_index (int): The index of the node within its layer.
        downstream (list): List of connections to downstream nodes (next layer).
        upstream (list): List of connections from upstream nodes (previous layer).
        output (float): The output value of the node after activation.
        delta (float): The delta value used for backpropagation.

    Methods:
        set_output(output):
            Sets the output value of the node.

        append_downstream_connection(conn):
            Adds a connection to the downstream list.

        append_upstream_connection(conn):
            Adds a connection to the upstream list.

        calc_output():
            Calculates the output of the node using the sigmoid activation function
            and the outputs of upstream nodes.

        calc_hidden_layer_delta():
            Calculates the delta value for a hidden layer node during backpropagation.

        calc_output_layer_delta(label):
            Calculates the delta value for an output layer node during backpropagation,
            based on the target label.

        __str__():
            Returns a string representation of the node, including its output, delta,
            and its upstream and downstream connections.
    """
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        self.output = output

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        self.upstream.append(conn)

    def calc_output(self):
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0)
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str


class ConstNode(object):
    """
    ConstNode represents a constant node (typically a bias node) in a neural network layer.

    Attributes:
        layer_index (int): The index of the layer this node belongs to.
        node_index (int): The index of the node within its layer.
        downstream (list): List of downstream connections from this node.
        output (float): The output value of the node, always set to 1 for constant nodes.
        delta (float): The delta value used during backpropagation.

    Methods:
        append_downstream_connection(conn):
            Adds a connection to the downstream list.

        calc_hidden_layer_delta():
            Calculates the delta for this node based on downstream nodes and their weights.

        __str__():
            Returns a string representation of the node, including its output and downstream connections.
    """
    def __init__(self, layer_index, node_index):
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1
        self.delta = 0

    def append_downstream_connection(self, conn):
        self.downstream.append(conn)

    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str


class Layer(object):
    """
    Represents a layer in a neural network, containing multiple nodes (neurons).

    Attributes:
        layer_index (int): The index of this layer in the network.
        nodes (list): A list of Node objects in this layer, including a ConstNode for bias.

    Methods:
        __init__(layer_index, node_count):
            Initializes the layer with the specified number of nodes and a bias node.
        set_output(data):
            Sets the output values for the nodes in this layer (excluding the bias node).
        calc_output():
            Calculates the output for each node in this layer (excluding the bias node).
        dump():
            Prints the string representation of each node in this layer.
    """
    def __init__(self, layer_index, node_count):
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        for node in self.nodes:
            print(node)


class Connection(object):
    """
    Represents a connection (synapse) between two nodes in a neural network.

    Attributes:
        upstream_node: The node from which the connection originates.
        downstream_node: The node to which the connection leads.
        weight (float): The weight of the connection, initialized randomly.
        gradient (float): The gradient of the connection, used for weight updates.

    Methods:
        calc_gradient():
            Calculates the gradient of the connection based on the downstream node's delta and the upstream node's output.

        update_weight(rate):
            Updates the connection's weight using the calculated gradient and the specified learning rate.

        get_gradient():
            Returns the current gradient of the connection.

        __str__():
            Returns a string representation of the connection, including node indices and weight.
    """
    def __init__(self, upstream_node, downstream_node):
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    def calc_gradient(self):
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def update_weight(self, rate):
        self.calc_gradient()
        self.weight += rate * self.gradient

    def get_gradient(self):
        return self.gradient

    def __str__(self):
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)


class Connections(object):
    """
    A class to manage and store a collection of connections, typically used in neural network implementations.

    Attributes:
        connections (list): A list to store connection objects.

    Methods:
        add_connection(connection):
            Adds a connection object to the connections list.

        dump():
            Prints all stored connection objects to the standard output.
    """
    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

    def dump(self):
        for conn in self.connections:
            print(conn)


class Network(object):
    """
    A simple feedforward neural network implementation supporting backpropagation training.

    Attributes:
        connections (Connections): Manages all connections (weights) between nodes in the network.
        layers (list of Layer): List of layers in the network, each containing nodes.

    Args:
        layers (list of int): List specifying the number of nodes in each layer.

    Methods:
        train(labels, data_set, rate, epoch):
            Trains the network on the provided dataset for a given number of epochs.

        train_one_sample(label, sample, rate):
            Trains the network on a single sample and its label.

        calc_delta(label):
            Calculates the delta (error term) for each node in the network using backpropagation.

        update_weight(rate):
            Updates the weights of all connections in the network based on calculated deltas.

        calc_gradient():
            Calculates the gradient of each connection for use in optimization.

        get_gradient(label, sample):
            Performs a forward and backward pass to compute gradients for a single sample.

        predict(sample):
            Performs a forward pass to compute the network's output for a given input sample.

        dump():
            Prints the output of each layer for debugging or inspection.
    """
    def __init__(self, layers):
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0;
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    def train(self, labels, data_set, rate, epoch):
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)
                # print 'sample %d training finished' % d

    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_delta(self, label):
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self, label, sample):
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self, sample):
        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])

    def dump(self):
        for layer in self.layers:
            layer.dump()


class Normalizer(object):
    """
    Normalizer provides methods to encode integers as normalized binary vectors and decode them back.

    Attributes:
        mask (list of int): List of bit masks for 8 bits (from least to most significant).

    Methods:
        norm(number):
            Converts an integer into a list of 8 normalized float values (0.9 for bit set, 0.1 for bit unset).
            Args:
                number (int): The integer to normalize.
            Returns:
                map: A map object yielding normalized float values for each bit.

        denorm(vec):
            Converts a list of normalized float values back into the original integer.
            Args:
                vec (iterable of float): Normalized float values (expected length: 8).
            Returns:
                int: The reconstructed integer from the normalized vector.
    """
    def __init__(self):
        self.mask = [
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
        ]

    def norm(self, number):
        return map(lambda m: 0.9 if number & m else 0.1, self.mask)

    def denorm(self, vec):
        binary = list(map(lambda i: 1 if i > 0.5 else 0, vec))
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x, y: x + y, binary)


def mean_square_error(vec1, vec2):
    """
    Calculates the mean squared error (MSE) between two vectors.

    Args:
        vec1 (iterable): The first vector of numerical values (e.g., predictions).
        vec2 (iterable): The second vector of numerical values (e.g., targets).

    Returns:
        float: The mean squared error between the two vectors, scaled by 0.5.

    Note:
        The function computes 0.5 * sum((v1 - v2)^2 for v1, v2 in zip(vec1, vec2)).
    """
    return 0.5 * reduce(lambda a, b: a + b,
                        map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                            zip(vec1, vec2)
                            )
                        )


def gradient_check(network, sample_feature, sample_label):
    """
    Performs gradient checking on a neural network to verify the correctness of backpropagation gradients.

    Args:
        network: The neural network object. Must implement `get_gradient(sample_feature, sample_label)`, 
                 `predict(sample_feature)`, and have a `connections.connections` iterable of connection objects.
        sample_feature: The input features for a single sample.
        sample_label: The true label for the sample.

    Description:
        For each connection weight in the network, this function perturbs the weight by a small value (epsilon)
        and computes the numerical gradient of the network error with respect to that weight. It then compares
        this numerical gradient (expected gradient) to the gradient computed by backpropagation (actual gradient)
        for each connection, printing both values for inspection.

    Note:
        This function is intended for debugging and validation purposes, and should not be used during normal training.
    """
     # Compute gradients using backpropagation
    network_error = lambda vec1, vec2: \
        0.5 * reduce(lambda a, b: a + b,
                     map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                         zip(vec1, vec2)))

    network.get_gradient(sample_feature, sample_label)

    # For each connection, numerically estimate the gradient and compare
    for conn in network.connections.connections:
        actual_gradient = conn.get_gradient()

        # add a small perturbation to the weights
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)

        # Subtract a small perturbation from the weights
        conn.weight -= 2 * epsilon  # We just added it once, so we need to subtract 2 times here
        error2 = network_error(network.predict(sample_feature), sample_label)

        # calcolate expected gradient
        expected_gradient = (error2 - error1) / (2 * epsilon)

        print('expected gradient: \t%f\nactual gradient: \t%f' % (
            expected_gradient, actual_gradient))


def train_data_set():
    """
    Generates a dataset of normalized values and their corresponding labels.

    This function creates a list of normalized values using a Normalizer instance.
    For each value in the range 0 to 255 (inclusive) with a step of 8, a random integer
    between 0 and 255 is generated, normalized, and added to the dataset. The same
    normalized value is also used as the label.

    Returns:
        tuple: A tuple containing two lists:
            - labels (list): The list of normalized labels.
            - data_set (list): The list of normalized data points.
    """
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(0, 256, 8):
        n = normalizer.norm(int(random.uniform(0, 256)))
        data_set.append(n)
        labels.append(n)
    return labels, data_set


def train(network):
    """
    Trains the given neural network using a training dataset.

    This function retrieves the training labels and data set, then calls the network's
    train method with the data, a learning rate of 0.3, and 50 epochs.

    Args:
        network: An object representing the neural network to be trained. It must have a
            `train(labels, data_set, learning_rate, epochs)` method.

    Returns:
        None
    """
    labels, data_set = train_data_set()
    network.train(labels, data_set, 0.3, 50)


def test(network, data):
    """
    Tests a neural network by normalizing the input data, making a prediction, and printing both the original and predicted values.

    Args:
        network: The neural network object with a `predict` method.
        data: The input data to be tested.

    Returns:
        None. Prints the original and predicted values to the console.
    """
    normalizer = Normalizer()
    norm_data = normalizer.norm(data)
    predict_data = network.predict(norm_data)
    print('\ttestdata(%u)\tpredict(%u)' % (
        data, normalizer.denorm(predict_data)))


def correct_ratio(network):
    """
    Calculates and prints the percentage of correctly predicted values by the network.

    This function iterates over all integer values from 0 to 255, normalizes each value,
    feeds it to the network's predict method, then denormalizes the prediction.
    If the denormalized prediction matches the original value, it is counted as correct.
    Finally, the function prints the ratio of correct predictions as a percentage.

    Args:
        network: An object with a `predict` method that takes a normalized input and returns a prediction.

    Returns:
        None. Prints the correct prediction ratio as a percentage.
    """
    normalizer = Normalizer()
    correct = 0.0;
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print('correct_ratio: %.2f%%' % (correct / 256 * 100))


def gradient_check_test():
    """
    Tests the gradient checking implementation for a neural network.

    This function creates a simple neural network with an architecture of 2 input neurons,
    one hidden layer with 2 neurons, and 2 output neurons. It then defines a sample input
    feature vector and a corresponding label. The function calls `gradient_check` to verify
    the correctness of the network's backpropagation gradients by comparing them to numerically
    computed gradients.

    Returns:
        None
    """
    net = Network([2, 2, 2])
    sample_feature = [0.9, 0.1]
    sample_label = [0.9, 0.1]
    gradient_check(net, sample_feature, sample_label)


if __name__ == '__main__':
    net = Network([8, 3, 8])
    train(net)
    net.dump()
    correct_ratio(net)