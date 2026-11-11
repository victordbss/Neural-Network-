import ia as ia
from data import *

net = ia.Network(layers_sizes=[4, 10, 10, 2], learning_rate=0.5)

net.training(inputs=X, outputs=Y, training_duration=120)
net.save("data2")
#net.load("data2")

"""user_input = ""
while user_input != "stop":
    user_input = input("valeur a tester :")

    test = [float(i) for i in  user_input.split(',')]
    print(net.forward([test]))"""

