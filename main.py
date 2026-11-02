import ia as ia
from data import *

net = ia.Network(layers_sizes=[4, 10, 10, 2], learning_rate=1)
net.load("data1")
#net.training(X, Y, 30000, 100, 1.01)
#net.save("data1")

user_input = ""
while user_input != "stop":
    user_input = input("valeur a tester :")

    test = [int(i) for i in  user_input.split(',')]
    for i in net.forward([test])[0]:
        print(round(i))
