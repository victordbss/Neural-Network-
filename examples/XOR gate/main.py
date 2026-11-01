import sys
import ia as ia
from data import *

net = ia.Network(layers_sizes=[2, 4, 4, 1], learning_rate=1)
net.training(X, Y, 1000, 100, 1.01)


user_input = ""
while user_input != "stop":
    user_input = input("valeur a tester :")

    test = [int(i) for i in  user_input.split(',')]
    print(net.forward([test])[0])
