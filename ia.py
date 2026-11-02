from __future__ import annotations
import numpy as np
import json
import os

class InputLayer:
    def __init__(self, output_size : int):
        self.output_size = output_size
        self.node_value : list = []

class Layer:
    def __init__(self, input_layer : Layer | InputLayer , output_size : int, weights : np.array = np.array([]), biases : np.array = np.array([]), output_layer : bool = False):
        self.input_layer = input_layer
        self.output_size = output_size
        self.weights = weights
        self.biases = biases
        self.output_layer = output_layer
        self.nodes_value : list = []
        self.z : list = []
        self.init_layer()

    def init_layer(self):
        if self.weights.size == 0:
            self.init_weights()
        if self.biases.size == 0:
            self.init_biases()

    def init_weights(self):
        n_in = self.input_layer.output_size
        n_out = self.output_size

        limit = np.sqrt(6 / (n_in + n_out))

        self.weights = np.random.uniform(
            -limit,
            limit,
            (n_out, n_in)
        )

    def init_biases(self):
        self.biases = np.zeros(self.output_size)

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs : np.array):
        A = np.dot(self.weights, inputs) + self.biases
        self.z.append(A)
        if not self.output_layer:
            A = self.sigmoid(A)
        self.nodes_value.append(A)
        return A

    def reset(self):
        self.nodes_value = []
        self.z = []



class Network:
    def __init__(self, layers_sizes : list [int], learning_rate : float = 0.1):
        self.layers_sizes = layers_sizes
        self.layers : list [Layer | InputLayer] = []
        self.learning_rate = learning_rate
        self.previous_cost = None
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


    def forward(self, inputs_list):
        for layer in self.layers[1:]:
            layer.reset()

        self.layers[0].nodes_value = inputs_list
        for inputs in inputs_list:
            next_input = inputs
            for layer in self.layers[1:]:
                next_input = layer.forward(next_input)
        return np.array(self.layers[-1].nodes_value)
        

    def cost_function(self, inputs, outputs):
        return np.sum((self.forward(inputs) - outputs)**2) * (1/(2 * len(inputs)))

    def dsig(self, x):
        x = np.clip(x, -500, 500)
        xsig = 1 / (1 + np.exp(-x))
        return xsig * (1 - xsig)
 
    def back_propagation(self, inputs, outputs):
        vectorized_dsig = np.vectorize(self.dsig)

        # COUCHE SORTIE L
        # pas de dérivée puisque je veux que ma sortie puisse avoir des valeurs en dehors de 0, 1
        erreur_sortie = (1/len(outputs))*(np.array(self.layers[-1].nodes_value).T - outputs.T) 
        gradwL = np.dot(erreur_sortie, np.array(self.layers[-2].nodes_value))
        gradbL = erreur_sortie.sum(axis = 1)


        delta = [erreur_sortie]
        Weight_gradiant = [gradwL]
        Biases_gradiant = [gradbL]

        for i in range(len(self.layers_sizes) - 2):
            newdelta = np.dot(self.layers[-i-1].weights.T, delta[i]) * self.dsig(np.array(self.layers[-i-2].z).T)
            Weight_gradiant.append(np.dot(newdelta, np.array(self.layers[-i-3].nodes_value)))
            Biases_gradiant.append(newdelta.sum(axis = 1))
            delta.append(newdelta)




        layers = self.layers[1:][::-1]  

        old_weights = [layer.weights.copy() for layer in layers]
        old_biases = [layer.biases.copy() for layer in layers]

        old_cost = self.cost_function(inputs, outputs)

        grad_norm_sq = sum(
            np.sum(gw**2) + np.sum(gb**2)
            for gw, gb in zip(Weight_gradiant, Biases_gradiant)
        )

        eta = self.learning_rate
        beta = 0.5
        c = 1e-4

        accepted = False

        for _ in range(30):

            for i, layer in enumerate(layers):
                layer.weights = old_weights[i] - eta * Weight_gradiant[i]
                layer.biases = old_biases[i] - eta * Biases_gradiant[i]

            new_cost = self.cost_function(inputs, outputs)

            if np.isfinite(new_cost) and (
                new_cost <= old_cost - c * eta * grad_norm_sq
            ):
                accepted = True
                break

            eta *= beta

        if not accepted:
            for i, layer in enumerate(layers):
                layer.weights = old_weights[i]
                layer.biases = old_biases[i]


        

    def propagation(self, inputs, outputs):
        self.forward(inputs)
        self.back_propagation(inputs, outputs)

    def training(self, inputs, outputs, epoch, notify_rate, learning_rate_ratio, package_size):
        data_size = len(inputs)
        package_index = 0
        for i in range(epoch):
            intputs_pack = inputs[package_index*package_size:(package_index + 1)*package_size]
            outputs_pack = outputs[package_index*package_size:(package_index + 1)*package_size]
            self.propagation(intputs_pack, outputs_pack)

            if i % notify_rate == 0:
                cost = self.cost_function(intputs_pack, outputs_pack)
                print(f"Epoch {i} : coût = {cost:.6f}")
                self.learning_rate /= learning_rate_ratio

            if (package_index + 2) * package_size > data_size:
                package_index = 0
            else:
                package_index += 1

        print(f"TRAINING FINISHED AFTER {epoch} EPOCH")
        print(f"==================")
        cost = self.cost_function(inputs, outputs)
        print(f"-> COST : {cost:.6f}")

    def save(self, file_name : str):
        print("Saving ...")
        data = {
            "layers_sizes" : self.layers_sizes,
            "input_layer_size" : self.layers[0].output_size
        }

        for i, layer in enumerate(self.layers[1:]):
            data[i] = {
                "output_size" : layer.output_size,
                "weights" : layer.weights.tolist(),
                "biases" : layer.biases.tolist()
            }

        os.makedirs("data", exist_ok=True) 

        with open("data/" + file_name, "w") as f:
            json.dump(data, f, indent=4)

        print("Save finished !")

    def load(self, filename : str):
        with open("data/" + filename, "r") as f:
            data = json.load(f)
        self.layers_sizes = data['layers_sizes']
        self.layers = [InputLayer(output_size=data['input_layer_size'])]
        for i in range(len(self.layers_sizes[1:])):
            layer = data[str(i)]
            self.layers.append(Layer(input_layer=self.layers[i], output_size=layer['output_size'], weights=np.array(layer['weights']), biases=np.array(layer['biases'])))
        self.layers[-1].output_layer = True

        



