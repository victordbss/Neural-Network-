from __future__ import annotations
import numpy as np
import math
from data import *


class InputLayer:
    def __init__(self, output_size):
        self.output_size = output_size

class Layer:
    def __init__(self, input_layer : Layer | InputLayer , output_size : int, weights : np.array = None, biases : np.array = None, output_layer : bool = False):
        self.input_layer = input_layer
        self.output_size = output_size
        self.weights = weights
        self.biases = biases
        self.output_layer = output_layer
        self.nodes_value : np.array = None
        self.init_layer()

    def init_layer(self):
        if not self.weights:
            self.init_weights()
        if not self.biases:
            self.init_biases()

    def init_weights(self):
        self.weights = np.random.random((self.output_size, self.input_layer.output_size))

    def init_biases(self):
        self.biases = np.zeros(self.output_size)

    def sigmoid(self, x):
        return 1/(1+ math.exp(-x))

    def forward(self, inputs : np.array):
        next_input = np.zeros(self.output_size)
        A = np.dot(self.weights, inputs) + self.biases
        if not self.output_layer:
            vectorized_sigmoid = np.vectorize(self.sigmoid)
            A = vectorized_sigmoid(A)
        self.nodes_value = A
        return A


class Network:
    def __init__(self, layers_sizes : list [int], learning_rate : float = 0.1):
        self.layers_sizes = layers_sizes
        self.layers : list [Layer | InputLayer] = []
        self.learning_rate = learning_rate
        self.init_layers()

    def init_layers(self):
        self.layers.append(InputLayer(output_size=self.layers_sizes[0]))
        for i, l_size in enumerate(self.layers_sizes[1:]):
            self.layers.append(Layer(input_layer=self.layers[i], output_size=l_size))
        self.layers[-1].output_layer = True

    def shape(self):
        res = ""
        for l_size in self.layers_sizes:
            res += f"{l_size} - "
        print(res[:-2])
        for i, layer in enumerate(self.layers[1:]):
            print(f"### Layer {i + 1} ###")
            print(f"Weights = {layer.weights}")
            print(f"Biases = {layer.biases}")

    def forward(self, inputs):
        for layer in self.layers[1:]:
            inputs = layer.forward(inputs)
        return inputs

    def cost_function(self, inputs, outputs):
        cost_vect = 0
        for i, input in enumerate(inputs):
            cost_vect += (self.forward(inputs[i]) - outputs[i]) ** 2
        cost_vect /= (len(inputs) * len(cost_vect))
        cost = np.sum(cost_vect)
        return cost

    def sigmoide(self, x):
        return 1/(1+math.exp(-x))

    def dsig(self, x):
        xsig = self.sigmoide(x)
        return xsig*(1-xsig)

    # ATENTION dans le cas 1 hidden layer
    
    def back_propagation(self, inputs, outputs):
        for i, input_v in enumerate(inputs):

            prediction = self.forward(input_v)

            hidden_layer = self.layers[-2]
            output_layer = self.layers[-1]

            H = hidden_layer.nodes_value

            # Couche de sortie (linéaire)
            delta2 = prediction - outputs[i]

            gradweights2 = np.outer(delta2, H)
            gradbiases2 = delta2

            # Couche cachee (sigmoide)
            delta1 = (output_layer.weights.T @ delta2) * H * (1 - H)

            gradweights1 = np.outer(delta1, input_v)
            gradbiases1 = delta1
            
            output_layer.weights -= self.learning_rate * gradweights2
            output_layer.biases -= self.learning_rate * gradbiases2

            hidden_layer.weights -= self.learning_rate * gradweights1
            hidden_layer.biases -= self.learning_rate * gradbiases1







net = Network(layers_sizes=[4, 25, 2], learning_rate=0.0001)

for epoch in range(100000):
    indices = np.random.permutation(len(X))

    net.back_propagation(X[indices], Y[indices])

    if epoch % 100 == 0:
        cost = net.cost_function(X, Y)
        print(f"Epoch {epoch} : coût = {cost:.6f}")
        net.learning_rate /= 1.0005

    
print(net.cost_function(X, Y))

user_input = ""
while user_input != "stop":
    user_input = input("valeur a tester :")

    test = [int(i) for i in  user_input.split(',')]
    print([round(i) for i in net.forward(test)])


