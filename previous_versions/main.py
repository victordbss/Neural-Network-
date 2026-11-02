import numpy as np
import random

CONST = 0.01

w1, w2 = random.random(), random.random()
b = random.random()

inputs = np.array([[1,0], [0,1], [1,1], [0,0]])
outputs = np.array([0.5,0.5,1,0])

"""
def sigmoid(x):
    return 1/(1 + np.exp(-x))
"""
    
def predict(w1,w2,b, i1, i2):
    return i1 * w1 + i2 * w2 + b

def cost_function(w1,w2,b, inputs, outputs):
    cost = 0
    for i, input in enumerate(inputs):
        cost += (predict(w1, w2, b, input[0], input[1]) - outputs[i])**2
    return cost

cost = cost_function(w1, w2, b, inputs, outputs)
print("### INIT VALUES ###")
print(f"w1 = {w1} ; w2 = {w2} ; b = {b}")
print(f"cost = {cost}")

for gen in range(100):
    print(f"### GEN {gen + 1} ###")

    gw1 = 0
    gw2 = 0
    gb = 0
    for i, input in enumerate(inputs):
        gw1 += input[0] * 2 * (predict(w1, w2, b, input[0], input[1]) - outputs[i])
        gw1 += input[1] * 2 * (predict(w1, w2, b, input[0], input[1]) - outputs[i])
        gb += 2 * (predict(w1, w2, b, input[0], input[1]) - outputs[i])

    gw1 /= len(outputs)
    gw2 /= len(outputs)
    gb /= len(outputs)

    print(f"gw1 = {gw1} ; gw2 = {gw2} ; gb = {gb}")

    w1 -= gw1 * CONST
    w2 -= gw2 * CONST
    b -= gb * CONST

    newcost = cost_function(w1, w2, b, inputs, outputs)
    print(f"w1 = {w1} ; w2 = {w2} ; b = {b}")
    print(f"New cost : {newcost}")


    