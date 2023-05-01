import numpy as np
import pandas as pd

class Neuron:
    def __init__(self, n_inputs=2, activation='tanh'):
        self.w = np.random.randn(n_inputs) * 0.01
        self.b = np.random.randn() * 0.01
        self.activation_type = activation
        self.output = None
        self.inputs = None

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
        output = self.activation(np.dot(self.w, X) + self.b)
        self.inputs, self.output = X, output
        return output
    

def forward_layer(neurons, inputs):
    return np.array([neuron.forward(inputs) for neuron in neurons])


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
    def __init__(self, input_layer, hidden_layers, output_layer, loss='binary_cross_entropy'):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.loss = loss

    def calculate_loss(self, predictions, y):
        if self.loss == 'binary_cross_entropy':
            return [-y * np.log(prediction) - (1 - y) * np.log(1 - prediction) for prediction in predictions]
    
    def loss_derivative(self, prediction, y):
        if self.loss == 'binary_cross_entropy':
            return -y / prediction + (1 - y) / (1 - prediction)

    def backpropagation(self, X, y, predictions, learning_rate):
        
        # calculate gradients and update output layer parameters
        for neuron, prediction in zip(self.output_layer.neurons, predictions):
            bias_gradient = neuron.activation_derivative(prediction) * self.loss_derivative(prediction, y)
            weight_gradients = [x * bias_gradient for x in neuron.inputs]
            neuron.w -= learning_rate * np.array(weight_gradients)
            neuron.b -= learning_rate * bias_gradient
        

    def train(self, X, y, epochs=10, learning_rate=0.05):
        for epoch in range(epochs):
            losses = np.array([])
            for Xi, yi in zip(X, y):
                predictions, loss = self.forward_pass(Xi, yi)
                losses = np.append(losses, loss)
                self.backpropagation(Xi, yi, predictions, learning_rate)
            if epoch < 10 or epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {losses.mean()}")

    def forward_pass(self, X, y=None):
        input_layer_output = self.input_layer.forward(X)
        hidden_layers_output = self.hidden_layers.forward(input_layer_output)
        predictions = self.output_layer.forward(hidden_layers_output)
        losses = self.calculate_loss(predictions, y) if y is not None else None
        return predictions, losses
    

input_layer = InputLayer(n_inputs=2, activation='tanh')
hidden_layers = HiddenLayers(prev_layer=input_layer, height=3, depth=2, activation='tanh')
output_layer = OutputLayer(prev_layer=hidden_layers, activation='sigmoid')

model = NeuralNetwork(input_layer, hidden_layers, output_layer)

# prediction, loss = model.forward_pass(np.array([1, 2]), 1)
# print(prediction, loss)

df = pd.read_csv('fluffy_or_spikey.csv')
X, y = df[['height', 'color']].to_numpy(), df['species'].to_numpy()

model.train(X, y, 2)