from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import json
import math
from os import makedirs
import time

"""
La classe InputLayer est toujours le premier élèment de la liste des Layers présents dans la classe Network
Elle représente les entrées injectées dans le réseau tout en gardant les mêmes noms de variable que la classe Layer 
pour facilité les opérations sans faire de distinction entre ces deux classes.
"""
class InputLayer:
    def __init__(self, output_size : int = 1):
        self.output_size = output_size
        self.node_value : list[float] = []

"""
La classe layer correspond, dans le réseau, aux neurones de la couches d'indice l et les connexions avec la couche l-1
Elle reprend donc les valeurs des neurones de la couche l ainsi que la matrice poids et biais qui permet de passer de la couche précédente
à la couche actuelle
"""
class Layer:
    def __init__(self, input_layer : Layer | InputLayer , output_size : int, weights : np.array = np.array([]), biases : np.array = np.array([]), output_layer : bool = False):
        self.input_layer = input_layer
        self.output_size = output_size

        self.weights = weights
        self.biases = biases

        self.output_layer = output_layer # Vraie seulement si la couche est la dernière du réseau (celle qui génère l'output)

        self.nodes_value : np.array = np.array([]) # Valeur des neurones après être passer dans la fonction d'activation (sigmoide)
        self.z : np.array = np.array([]) # Valeur des neuronnes avant d'être passer dans la fonction d'activation
        self.init_layer()

    def init_layer(self):
        if self.weights.size == 0:
            self.init_weights()
        if self.biases.size == 0:
            self.init_biases()

    # Initie les valeurs de la matrice poids en fonction des tailles des couches
    def init_weights(self):
        n_in = self.input_layer.output_size
        n_out = self.output_size

        limit = np.sqrt(6 / (n_in + n_out))

        self.weights = np.random.uniform(
            -limit,
            limit,
            (n_out, n_in)
        )

    # Initie le vecteur biais à 0
    def init_biases(self):
        self.biases = np.zeros(self.output_size)

    # Fonction d'activation
    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    # Génère la valeur des neurones de la couche l en fonction des valeurs de la couche l-1
    def forward(self, inputs : np.array):
        A = np.dot(inputs, self.weights.T) + self.biases
        self.z = A
        if not self.output_layer:
            A = self.sigmoid(A)
        self.nodes_value = A
        return A




"""
Classe principale du réseau
Elle comprends la tailles des couches de celui-ci ainsi que la liste des Layers
Elle fait la liaison entre toutes les couches
"""
class Network:
    def __init__(self, layers_sizes : list [int], learning_rate : float = 0.1):
        self.layers_sizes = layers_sizes
        self.layers : list [Layer | InputLayer] = []

        self.learning_rate = learning_rate

        self.previous_cost = None

        self.init_layers()

    # Initie la liste de Layer en fonction de la taille du neuronnes (taille des couches)
    def init_layers(self):
        self.layers.append(InputLayer(output_size=self.layers_sizes[0]))
        
        for i, l_size in enumerate(self.layers_sizes[1:]):
            self.layers.append(Layer(input_layer=self.layers[i], output_size=l_size))

        self.layers[-1].output_layer = True


    # A modifier
    # Fonction qui donnes des infos sur la tailles du neuronnes ainsi que ses poids et biais
    def shape(self):
        res = ""
        for l_size in self.layers_sizes:
            res += f"{l_size} - "
        print(res[:-2])
        for i, layer in enumerate(self.layers[1:]):
            print(f"### Layer {i + 1} ###")
            print(f"Weights = {layer.weights}")
            print(f"Biases = {layer.biases}")

    # Fonction qui transmet les inputs à travers le réseau jusqu'à l'output
    def forward(self, inputs_list : np.array) -> np.array:

        self.layers[0].nodes_value = inputs_list
        # Fait passer input par input à travers le réseau
        
        next_input = inputs_list
        for layer in self.layers[1:]:
            next_input = layer.forward(next_input)

        return next_input
        
    # Fonction de coût (mesure l'erreur du réseau face à la matrice inputs)
    def cost_function(self, inputs : np.array, outputs : np.array) -> float:
        return np.sum((self.forward(inputs) - outputs)**2) * (1/(2 * len(inputs)))

    # Fonction dérivée de la fonction d'activation (sigmoide)
    def dsig(self, x : np.array) -> np.array:
        x = np.clip(x, -500, 500)
        xsig = 1 / (1 + np.exp(-x))
        return xsig * (1 - xsig)

    # Fonction qui vient modifier les poids et biais du réseau pour faire diminuer la valeur de la fonction coût
    def back_propagation(self, inputs : np.array, outputs : np.array) -> tuple:

        # COUCHE SORTIE L
        # pas de dérivée puisque je veux que ma sortie puisse avoir des valeurs en dehors de 0, 1
        erreur_sortie = (1/len(outputs))*(self.layers[-1].nodes_value.T - outputs.T) 
        gradwL = np.dot(erreur_sortie, self.layers[-2].nodes_value) #gardiant des poids de la dernière couche
        gradbL = erreur_sortie.sum(axis = 1) #gradiant des biais de la dernière couche

        # Le delta est une partie du calcul du gradiant de la couche l qui doit être stocké car il est nécessaire au calcul du gradiant
        # pour la couche l-1
        delta = [erreur_sortie]

        Weight_gradiant = [gradwL]
        Biases_gradiant = [gradbL]

        for i in range(len(self.layers_sizes) - 2):
            newdelta = np.dot(self.layers[-i-1].weights.T, delta[i]) * self.dsig(self.layers[-i-2].z.T)
            Weight_gradiant.append(np.dot(newdelta, self.layers[-i-3].nodes_value))
            Biases_gradiant.append(newdelta.sum(axis = 1))
            delta.append(newdelta)

        # On ne modifie pas directement les poids et biais en fonction de leur gradiant car si le learning rate n'est pas adapté alors
        # la valeur de la fonction coût pourrait augmenter.
        # On va donc regarder le coût de la fonction avant la modification et la comparée avec le cout après modification
        # Si le coût est supérieur alors on diminue le learning rate

        return Weight_gradiant, Biases_gradiant

    def update_parameters(self, inputs : np.array, outputs : np.array, Weight_gradiant : list, Biases_gradiant : list):
        layers = self.layers[1:][::-1]  
        
        old_weights = [layer.weights.copy() for layer in layers]
        old_biases = [layer.biases.copy() for layer in layers]

        old_cost = self.cost_function(inputs=inputs, outputs=outputs)

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

            new_cost = self.cost_function(inputs=inputs, outputs=outputs)

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


    # Fonction qui propage l'information dans le réseau en avant (forward) puis repart en arrière pour modifier les paramètres (back_propagation)
    def propagation(self, inputs : np.array, outputs : np.array):
        self.forward(inputs)
        Weight_gradiant, Biases_gradiant = self.back_propagation(inputs=inputs, outputs=outputs)
        self.update_parameters(inputs=inputs, outputs=outputs, Weight_gradiant=Weight_gradiant, Biases_gradiant=Biases_gradiant)



    # Fonction qui permet d'entrainer le neurones pendant un certains nombre de générations
    def training(self, inputs : np.array, outputs : np.array, epoch : int = None, training_duration : float = None, notify_rate : int = 100, learning_rate_ratio : float = 1.001, batch_size : int = None, graph : bool = False):
        data_size = len(inputs)

        if not batch_size:
            batch_size = data_size

        if not epoch:
            epoch = 10**10

        start_time = time.time()
        for i in range(epoch):
            indices = np.random.permutation(data_size)
            X = inputs[indices]
            Y = outputs[indices]
            for start in range(0, data_size, batch_size):
                x_batch = X[start:start + batch_size]
                y_batch = Y[start:start + batch_size]
                self.propagation(x_batch, y_batch)

            if i % notify_rate == 0:
                cost = self.cost_function(inputs=X, outputs=Y)
                print(f"Epoch {i} : coût = {cost:.6f}")
                self.learning_rate /= learning_rate_ratio
                if training_duration:
                    if time.time() - start_time >= training_duration:
                        break

                if graph:
                    plt.plot(i, math.log10(cost), "or")
        if graph:
            plt.xlabel("Epochs")
            plt.ylabel("Cost function")
            plt.show()


            

        print(f"TRAINING FINISHED AFTER {i} EPOCH and {round(time.time() - start_time, 3)} seconds")
        print(f"==================")
        cost = self.cost_function(inputs=inputs, outputs=outputs)
        print(f"-> COST : {cost:.6f}")

    # Fonction qui permet de sauvegarder l'état du réseau dans un fichier json
    def save(self, file_name : str = "new_data"):
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

        makedirs("data", exist_ok=True) # Création d'un dossier si il n'existe pas

        with open("data/" + file_name, "w") as f:
            json.dump(data, f, indent=4)

        print("Save finished !")

    # Fonction qui permet d'importer l'état d'un réseau
    def load(self, filename : str = "new_data"):

        with open("data/" + filename, "r") as f:
            data = json.load(f)

        self.layers_sizes = data['layers_sizes']
        self.layers = [InputLayer(output_size=data['input_layer_size'])]
        
        for i in range(len(self.layers_sizes[1:])):
            layer = data[str(i)]
            self.layers.append(Layer(input_layer=self.layers[i], output_size=layer['output_size'], weights=np.array(layer['weights']), biases=np.array(layer['biases'])))
        self.layers[-1].output_layer = True

        



