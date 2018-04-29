import NeuralNet
import generate_data
import numpy as np
import matplotlib.pyplot as mp

# tabulates all 2 bit inputs and computes the XOR value
X = generate_data.tabulate(3)
Y = generate_data.get_label(X)

# creates a neural network with 1 hidden layer
structure = (3,2)
xor_network = NeuralNet.dnn(X.shape[0],
                            Y.shape[0],
                            structure,
                            1)
print(X)
print(Y)
##print(xor_network.hidden_units[0])
##print(xor_network.hidden_units[1])
##print(xor_network.hidden_units[1])
##print(xor_network.output_layer)
##prediction = xor_network.predict(X)
##print(prediction)
##
loss = xor_network.learn(X,Y,0.005,15000)
##print(xor_network.hidden_units[0])
##print(xor_network.hidden_units[1])
##print(xor_network.hidden_units[1])
##print(xor_network.output_layer)
prediction = xor_network.predict(X)
print(prediction)

mp.ylim(0,max(loss)+1)
mp.plot(loss)
mp.show()

