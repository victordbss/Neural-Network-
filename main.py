import ia as ia
from data import *

net = ia.Network(layers_sizes=[4, 10, 10, 2], learning_rate=0.1)
net.training(X, Y, 1000000, 100, 1.001)


user_input = ""
while user_input != "stop":
    user_input = input("valeur a tester :")

    test = [int(i) for i in  user_input.split(',')]
    print(net.forward([test])[0])
