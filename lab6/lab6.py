# MIT 6.034 Lab 6: Neural Nets
# Written by 6.034 Staff

from nn_problems import *
from math import e
INF = float('inf')


#### Part 1: Wiring a Neural Net ###############################################

nn_half = [1]

nn_angle = [2,1]

nn_cross = [2,2,1]

nn_stripe = [3,1]

nn_hexagon = [6,1]

nn_grid = [4,2,1]


#### Part 2: Coding Warmup #####################################################

# Threshold functions
def stairstep(x, threshold=0):
    "Computes stairstep(x) using the given threshold (T)"
    return x >= threshold

def sigmoid(x, steepness=1, midpoint=0):
    "Computes sigmoid(x) using the given steepness (S) and midpoint (M)"
    return 1/(1+e**(-steepness*(x-midpoint)))

def ReLU(x):
    "Computes the threshold of an input using a rectified linear unit."
    return max([0,x])

# Accuracy function
def accuracy(desired_output, actual_output):
    "Computes accuracy. If output is binary, accuracy ranges from -0.5 to 0."
    return -.5*(desired_output-actual_output)**2


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
    node_dict = {}
    output = None
    for node in net.topological_sort():

        total = 0
        for input_val_name in net.get_incoming_neighbors(node):
            input_val = input_val_name
            if input_val_name in input_values:
                input_val = input_values[input_val_name]
            elif input_val_name in node_dict:
                input_val = node_dict[input_val_name]
            for wire in net.get_wires(input_val_name, node):
                total += input_val * wire.get_weight()
        node_dict[node] = threshold_fn(total)
        if net.is_output_neuron(node):
            output =threshold_fn(total)

    return (output, node_dict)



#### Part 4: Backward Propagation ##############################################

def gradient_ascent_step(func, inputs, step_size):
    """Given an unknown function of three variables and a list of three values
    representing the current inputs into the function, increments each variable
    by +/- step_size or 0, with the goal of maximizing the function output.
    After trying all possible variable assignments, returns a tuple containing:
    (1) the maximum function output found, and
    (2) the list of inputs that yielded the highest function output."""
    answer = func(inputs[0], inputs[1], inputs[2])
    answer_inputs = inputs
    options = [0,-step_size,step_size]
    for zero in options:
        for one in options:
            for two in options:
                if func(inputs[0]+zero, inputs[1]+one, inputs[2]+two) > answer:
                    answer = func(inputs[0]+zero, inputs[1]+one, inputs[2]+two)
                    answer_inputs = [inputs[0]+zero, inputs[1]+one, inputs[2]+two]

    return (answer, answer_inputs)

def get_back_prop_dependencies(net, wire):
    """Given a wire in a neural network, returns a set of inputs, neurons, and
    Wires whose outputs/values are required to update this wire's weight."""
    dependencies = [wire.startNode, wire, wire.endNode]
    for other_wire in net.get_wires(wire.endNode):
        if other_wire is not wire:
            dependencies.extend(get_back_prop_dependencies(net, other_wire))
    return set(dependencies)

def calculate_deltas(net, desired_output, neuron_outputs):
    """Given a neural net and a dictionary of neuron outputs from forward-
    propagation, computes the update coefficient (delta_B) for each
    neuron in the net.  Uses the sigmoid function to compute neuron output.
    Returns a dictionary mapping neuron names to update coefficient (the
    delta_B values). """
    delta = {}
    for neuron in net.topological_sort()[::-1]:
        if net.is_output_neuron(neuron):
            delta[neuron] = neuron_outputs[neuron]*(1-neuron_outputs[neuron])*(desired_output-neuron_outputs[neuron])
        else:
            delta[neuron] = 0
            for neighbor in net.get_outgoing_neighbors(neuron):
                delta[neuron] += neuron_outputs[neuron] * (1-neuron_outputs[neuron]) * net.get_wires(neuron, neighbor)[0].get_weight()*delta[neighbor]
    return delta

def update_weights(net, input_values, desired_output, neuron_outputs, r=1):
    """Performs a single step of back-propagation.  Computes delta_B values and
    weight updates for entire neural net, then updates all weights.  Uses the
    sigmoid function to compute neuron output.  Returns the modified neural net,
    with the updated weights."""
    delta = calculate_deltas(net, desired_output, neuron_outputs)
    weight = {}
    for wire in net.get_wires():
        output = wire.startNode
        if output in input_values:
            output = input_values[output]
        elif output in neuron_outputs:
            output = neuron_outputs[output]
        weight[wire] =  wire.get_weight() + r * output * delta[wire.endNode]
    for wire in net.get_wires():
        wire.set_weight(weight[wire])
    return net

def back_prop(net, input_values, desired_output, r=1, minimum_accuracy=-0.001):
    """Updates weights until accuracy surpasses minimum_accuracy.  Uses the
    sigmoid function to compute neuron output.  Returns a tuple containing:
    (1) the modified neural net, with trained weights
    (2) the number of iterations (that is, the number of weight updates)"""
    forward = forward_prop(net, input_values, sigmoid)
    output = forward[0]
    neuron_outputs = forward[1]
    iterations = 0
    while -.5 * (desired_output - output)**2 < minimum_accuracy:
        iterations += 1
        update_weights(net, input_values, desired_output, neuron_outputs, r)
        forward = forward_prop(net, input_values, sigmoid)
        output = forward[0]
        neuron_outputs = forward[1]
    return (net,iterations)

#### Part 5: Training a Neural Net #############################################

ANSWER_1 = 21
ANSWER_2 = 44
ANSWER_3 = 4
ANSWER_4 = 200
ANSWER_5 = 37

ANSWER_6 = 1
ANSWER_7 = "checkerboard"
ANSWER_8 = ["small", "medium", "large"]
ANSWER_9 = "B"

ANSWER_10 = "D"
ANSWER_11 = ["A", "C"]
ANSWER_12 = ["A", "E"]


#### SURVEY ####################################################################

NAME = "Katya Bezugla"
COLLABORATORS = "None"
HOW_MANY_HOURS_THIS_LAB_TOOK = 7
WHAT_I_FOUND_INTERESTING = "How simple neural nets actually are to program"
WHAT_I_FOUND_BORING = "None"
SUGGESTIONS = None
