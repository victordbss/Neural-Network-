import ia as ia
from data import *

net = ia.Network(layers_sizes=[2, 5, 1], learning_rate=0.5)

net.training(X, Y, 10000, 100, 1.0001, 1000)
net.save("data2")
#net.load("data2")

user_input = ""
while user_input != "stop":
    user_input = input("valeur a tester :")

    test = [int(i) for i in  user_input.split(',')]
    print(net.forward([test])[0][0])

