# MIT 6.034 Lab 7: Neural Nets
# Written by 6.034 Staff

from nn_problems import *
from math import e
INF = float('inf')


#### Part 1: Wiring a Neural Net ###############################################

nn_half = [1] # one line single neuron that meets above a certain threshold, y > T

nn_angle = [2,1] # a line is defined by neuron x + y > T, combine two of these with single node

nn_cross = [2,2,1] # 2 lines, 2 regions, do an AND

nn_stripe = [3,1] # 3 lines, do a big OR statement 

nn_hexagon = [6,1] # 6 lines, all OR together

nn_grid = [4,2,1] # 4 lines, 2 distinct regions intersected


#### Part 2: Coding Warmup #####################################################

# Threshold functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    return int(x >= threshold)

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    return 1 / (1 + e ** (-steepness * (x - midpoint)))

def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    return max(0,x)

# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return -0.5 * (desired_output - actual_output) ** 2


#### Part 3: Forward Propagation ###############################################

def node_value(node, input_values, neuron_outputs):  # PROVIDED BY THE STAFF
    """
    Given 
     * a node (as an input or as a neuron),
     * a dictionary mapping input names to their values, and
     * a dictionary mapping neuron names to their outputs
    returns the output value of the node.
    This function does NOT do any computation; it simply looks up
    values in the provided dictionaries.
    """
    if isinstance(node, str):
        # A string node (either an input or a neuron)
        if node in input_values:
            return input_values[node]
        if node in neuron_outputs:
            return neuron_outputs[node]
        raise KeyError("Node '{}' not found in either the input values or neuron outputs dictionary.".format(node))
    
    if isinstance(node, (int, float)):
        # A constant input, such as -1
        return node
    
    raise TypeError("Node argument is {}; should be either a string or a number.".format(node))

def forward_prop(net, input_values, threshold_fn=stairstep):
    """Given a neural net and dictionary of input values, performs forward
    propagation with the given threshold function to compute binary output.
    This function should not modify the input net.  Returns a tuple containing:
    (1) the final output of the neural net
    (2) a dictionary mapping neurons to their immediate outputs"""

    """
    Strategy:
    - node_value gives the value of a node (based on input and nueron matched so far)
    - iterate through the neurons and update a nueron_outputs (order based on topological sort)
    - get the wires into the neurons via .get_wires(endNode=neuron)  
        - iterate through the wires, get the weight and take weighted sum
    - use threshold function to calculate nueron output and assign it in neuron_output
    - after all neuron assignment, net.get_output_neuron() to find net output
    """
    
    neurons = net.topological_sort()
    neuron_outputs = {}

    for neuron in neurons:
        incoming_wires = net.get_wires(endNode=neuron)
        
        x = 0
        for wire in incoming_wires:
            weight = wire.get_weight()
            x += weight * node_value(wire.startNode, input_values, neuron_outputs)
        
        neuron_outputs[neuron] = threshold_fn(x)

    net_output = neuron_outputs[net.get_output_neuron()]
    return net_output, neuron_outputs

#### Part 4: Backward Propagation ##############################################

def gradient_ascent_step(func, inputs, step_size):
    """Given an unknown function of three variables and a list of three values
    representing the current inputs into the function, increments each variable
    by +/- step_size or 0, with the goal of maximizing the function output.
    After trying all possible variable assignments, returns a tuple containing:
    (1) the maximum function output found, and
    (2) the list of inputs that yielded the highest function output."""
    
    """
    Strategy:
    - Make all 27 combinations, then find max based on function
    """
    # define the three functions
    def add_step_size (x): return x + step_size
    def subtract_step_size (x): return x - step_size
    def zero_step_size (x): return x

    step_size_fn = [add_step_size, subtract_step_size, zero_step_size]

    # iterate through all 27 combination for global max
    x, y, z = inputs

    max_output = -1 * INF
    max_input = []
    for f in step_size_fn:
        for g in step_size_fn:
            for h in step_size_fn:
                func_output = func(f(x), g(y), h(z))
                if func_output > max_output:
                    max_output = func_output
                    max_input = [f(x), g(y), h(z)]

    return max_output, max_input


def get_back_prop_dependencies(net, wire):
    """Given a wire in a neural network, returns a set of inputs, neurons, and
    Wires whose outputs/values are required to update this wire's weight."""
    
    """
    Strategy:
    - Keep a queue of the startNode, and work backwards until no more incoming neigbors
    - Start with wire's startNode
        - Get all incoming wires to this node using .get_wires(endNode=curr_node)
            - For each wire, add the wire itself and startNode (exclude endNode since already counted for)
            - Add each startNode to queue for further exploration
        - Queue terminates with input nodes which return empty list to .get_wires
    - Return the set
    """
    dependencies = {wire, wire.startNode, wire.endNode} 
    
    queue = [wire.startNode]
    while queue != []:
        curr_node = queue.pop(0)
        incoming_wires = net.get_wires(endNode=curr_node) 
        for wire in incoming_wires:
            dependencies.add(wire)
            dependencies.add(wire.startNode)
            queue.append(wire.startNode)

    return dependencies


def calculate_deltas(net, desired_output, neuron_outputs):
    """Given a neural net and a dictionary of neuron outputs from forward-
    propagation, computes the update coefficient (delta_B) for each
    neuron in the net. Uses the sigmoid function to compute neuron output.
    Returns a dictionary mapping neuron names to update coefficient (the
    delta_B values). """

    """ Use formula from specifications, starting from output node """
    neuron_deltas = {}

    neurons = net.topological_sort()
    neurons.reverse()
    for neuron in neurons:
        out = neuron_outputs[neuron]

        if net.is_output_neuron(neuron):
            delta = out * (1 - out) * (desired_output - out)
            neuron_deltas[neuron] = delta
        else:
            outgoing_delta_component = 0
            outgoing_wires = net.get_wires(startNode=neuron)
            for wire in outgoing_wires:
                outgoing_delta_component += wire.get_weight() * neuron_deltas[wire.endNode]
            delta = out * (1 - out) * outgoing_delta_component
            neuron_deltas[neuron] = delta

    return neuron_deltas

def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses the
    sigmoid function to compute neuron output.  Returns the modified neural net,
    with the updated weights."""
    raise NotImplementedError

def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses the
    sigmoid function to compute neuron output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""
    raise NotImplementedError


#### Part 5: Training a Neural Net #############################################

ANSWER_1 = None
ANSWER_2 = None
ANSWER_3 = None
ANSWER_4 = None
ANSWER_5 = None

ANSWER_6 = None
ANSWER_7 = None
ANSWER_8 = None
ANSWER_9 = None

ANSWER_10 = None
ANSWER_11 = None
ANSWER_12 = None


#### SURVEY ####################################################################

NAME = None
COLLABORATORS = None
HOW_MANY_HOURS_THIS_LAB_TOOK = None
WHAT_I_FOUND_INTERESTING = None
WHAT_I_FOUND_BORING = None
SUGGESTIONS = None
