import ia as ia
from data import *

net = ia.Network(layers_sizes=[2, 5, 1], learning_rate=0.5)

net.training(inputs=X, outputs=Y, training_duration=20, package_size=100)
net.save("data2")
#net.load("data2")

user_input = ""
while user_input != "stop":
    user_input = input("valeur a tester :")

    test = [int(i) for i in  user_input.split(',')]
    print(round(net.forward([test])[0][0]))

