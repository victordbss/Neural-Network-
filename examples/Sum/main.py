import ia as ia
from data import *

net = ia.Network(layers_sizes=[2, 5, 5, 1], learning_rate=0.5)

#net.training(inputs=X, outputs=Y, training_duration=30, batch_size=100, graph=True)
#net.save("data")
net.load("data")

user_input = ""
while user_input != "stop":
    user_input = input("valeur a tester :")

    test = [float(i) for i in  user_input.split(',')]
    print(net.forward([test])[0][0])

