import numpy as np
import pandas as pd

class Neuron:
    def __init__(self, n_inputs=2, activation='tanh'):
        self.w = np.random.randn(n_inputs) * 0.01
        self.b = np.random.randn() * 0.01,
        self.activation_type = activation

    def activation(self, n):
        if self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-n))
        elif self.activation_type == 'tanh':
            return np.tanh(n)

    def activation_derivative(self, n):
        if self.activation_type == 'sigmoid':
            return n * (1 - n)
        elif self.activation_type == 'tanh':
            return 1 - n ** 2

    def forward(self, X):
        return self.activation(np.dot(self.w, X) + self.b)
    

def forward_layer(neurons, inputs):
    return np.array([neuron.forward(inputs) for neuron in neurons])
    # return np.array([output for output in map(lambda neuron: neuron.forward(inputs), neurons)])


class InputLayer:
    def __init__(self, n_inputs, activation='tanh'):
        self.neurons = [Neuron(n_inputs, activation) for _ in range(n_inputs)]

    def forward(self, X):
        return forward_layer(self.neurons, X)
    
    
class HiddenLayers:
    def __init__(self, prev_layer, height, depth, activation='tanh'):
        self.depth =  depth
        self.layers = [[Neuron(len(prev_layer.neurons), activation) for _ in range(height)]] + [[Neuron(height, activation) for _ in range(height)] for _ in range(depth - 1)]

    def forward(self, inputs, layer=0):
        if layer == self.depth: return inputs
        return self.forward(
            forward_layer(self.layers[layer], inputs),
            layer + 1
        )


class OutputLayer:
    def __init__(self, prev_layer, height=1, activation='tanh'):
        self.neurons = [Neuron(len(prev_layer.layers[-1]), activation) for _ in range(height)]

    def forward(self, inputs):
        return forward_layer(self.neurons, inputs)
    

class NeuralNetwork:
    def __init__(self, input_layer, hidden_layers, output_layer):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

    def binary_cross_entropy_loss(self, prediction, y):
        return -y * np.log(prediction) - (1 - y) * np.log(1 - prediction)
    
    def backpropagation(X, y, prediction, learning_rate):
        pass

    def train (self, X, y, epochs=10, learning_rate=0.05):
        pass

    def forward_pass(self, X, y=None):
        input_layer_output = self.input_layer.forward(X)
        hidden_layers_output = self.hidden_layers.forward(input_layer_output)
        prediction = self.output_layer.forward(hidden_layers_output)
        loss = self.binary_cross_entropy_loss(prediction, y) if y is not None else None
        return prediction, loss
    

input_layer = InputLayer(n_inputs=2, activation='tanh')
hidden_layers = HiddenLayers(prev_layer=input_layer, height=3, depth=2, activation='tanh')
output_layer = OutputLayer(prev_layer=hidden_layers, activation='sigmoid')

model = NeuralNetwork(input_layer, hidden_layers, output_layer)

prediction, loss = model.forward_pass(np.array([1, 2]), 1)

print(prediction, loss)