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
        self.z : np.array = None
        self.nodes_value_multi : list[np.array] = []
        self.z_multi : list[np.array] = []
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
        self.z = A
        if not self.output_layer:
            vectorized_sigmoid = np.vectorize(self.sigmoid)
            A = vectorized_sigmoid(A)
        self.nodes_value = A
        return A

    def forward_multi(self, inputs : np.array):

        A = np.dot(self.weights, inputs) + self.biases
        self.z_multi.append(A)
        if not self.output_layer:
            vectorized_sigmoid = np.vectorize(self.sigmoid)
            A = vectorized_sigmoid(A)
        self.nodes_value_multi.append(A)
        return A

    def reset(self):
        self.nodes_value = None
        self.z = None



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

    def forward_data_pack(self, inputs_list):
        for layer in self.layers[1:]:
            layer.reset()

        for inputs in inputs_list:
            for layer in self.layers[1:]:
                layer.forward_multi(inputs)
        

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

    # Pour 1 input seulement    
    def back_propagation(self, inputs, outputs):
        index = 0
        self.forward_data_pack(inputs)
        vectorized_dsig = np.vectorize(self.dsig)

        outputs_matrix = []
        for output in outputs:
            outputs_matrix.append(np.array(output))
        
        outputs_matrix = np.array(outputs_matrix)

        print(outputs_matrix)
        print(np.array(self.layers[-1].nodes_value_multi).T)
        # COUCHE SORTIE L
        # pas de dérivée puisque je veux que ma sortie puisse avoir des valeurs en dehors de 0, 1
        erreur_sortie = (1/len(outputs))*(np.array(self.layers[-1].nodes_value_multi).T - outputs_matrix.T) 
        gradwL = np.dot(erreur_sortie, np.array(self.layers[-2].nodes_value_multi).T)
        gradbL = erreur_sortie.sum(axis = 1)
        print("step1")
        # COUCHE L-1
        
        delta1 = np.dot(self.layers[-1].weights.T, erreur_sortie) * vectorized_dsig(self.layers[-2].z.T)
        gradwL1 = np.dot(delta1, self.layers[-3].nodes_value.T)
        gradbL1 = delta1.sum(axis = 1)

        # COUCHE L-2

        delta2 = np.dot(self.layers[-2].weights.T, delta1) * vectorized_dsig(self.layers[-3].z.T)
        gradwL2 = np.dot(delta2, inputs[index].T)
        gradbL2 = delta2.sum(axis = 1)

        self.layers[-1].weights -= self.learning_rate * gradwL
        self.layers[-1].biases -= self.learning_rate * gradbL

        self.layers[-2].weights -= self.learning_rate * gradwL1
        self.layers[-2].biases -= self.learning_rate * gradbL1

        self.layers[-3].weights -= self.learning_rate * gradwL2
        self.layers[-3].biases -= self.learning_rate * gradbL2







net = Network(layers_sizes=[4, 4, 4, 2], learning_rate=0.01)

for epoch in range(1000):
    net.back_propagation(X, Y)
    print(net.cost_function(X, Y))
"""
    if epoch % 100 == 0:
        cost = net.cost_function(X, Y)
        print(f"Epoch {epoch} : coût = {cost:.6f}")
        net.learning_rate /= 1.0005
        """

    
print(net.cost_function(X, Y))

user_input = ""
while user_input != "stop":
    user_input = input("valeur a tester :")

    test = [int(i) for i in  user_input.split(',')]
    print([round(i) for i in net.forward(test)])


