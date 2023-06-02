import numpy as np
import pandas as pd

class Neuron:
    def __init__(self, n_inputs=2, activation='tanh'):
        self.w = np.random.randn(n_inputs) * 0.1
        self.b = np.random.randn() * 0.1
        self.activation_type = activation
        self.gradient = 0
        self.w_gradients = np.zeros(n_inputs)
        self.output = None
        self.inputs = None

    def activation(self, n):
        if self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-n))
        elif self.activation_type == 'tanh':
            return np.tanh(n)

    def activation_derivative(self):
        if self.activation_type == 'sigmoid':
            return self.output * (1 - self.output)
        elif self.activation_type == 'tanh':
            return 1 - self.output ** 2
        
    def compute_gradients(self, upstream_gradient):
        self.gradient = upstream_gradient * self.activation_derivative()
        self.w_gradients = self.gradient * self.inputs

    def update_parameters(self, learning_rate):
        self.w -= learning_rate * self.w_gradients
        self.b -= learning_rate * self.gradient

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

    def calculate_loss(self, prediction, y):
        if self.loss == 'binary_cross_entropy':
            return -y * np.log(prediction) - (1 - y) * np.log(1 - prediction)
        elif self.loss == 'mse':
            return np.mean((prediction - y)**2)
    
    def loss_derivative(self, prediction, y):
        if self.loss == 'binary_cross_entropy':
            return -y / prediction + (1 - y) / (1 - prediction)
        elif self.loss == 'mse':
            return 2 * (prediction - y)

    def backpropagation(self, y, predictions, learning_rate):

        for neuron, prediction in zip(self.output_layer.neurons, predictions):
            neuron.compute_gradients(self.loss_derivative(prediction, y))
            neuron.update_parameters(learning_rate)

        for layer_idx in reversed(range(self.hidden_layers.depth)):
            prev_layer = self.output_layer.neurons if layer_idx == self.hidden_layers.depth - 1 else self.hidden_layers.layers[layer_idx + 1]
            for neuron_idx, neuron in enumerate(self.hidden_layers.layers[layer_idx]):
                prev_layer_gradient = sum([prev_neuron.gradient * prev_neuron.w[neuron_idx] for prev_neuron in prev_layer])
                neuron.compute_gradients(prev_layer_gradient)
                neuron.update_parameters(learning_rate)

        for neuron_idx, neuron in enumerate(self.input_layer.neurons):
            prev_layer_gradient = sum([hidden_neuron.gradient * hidden_neuron.w[neuron_idx] for hidden_neuron in self.hidden_layers.layers[0]])
            neuron.compute_gradients(prev_layer_gradient)
            neuron.update_parameters(learning_rate)


    def train(self, X, y, epochs=10, learning_rate=0.05):
        for epoch in range(epochs):
            losses = np.array([])
            for Xi, yi in zip(X, y):
                predictions, loss = self.forward_pass(Xi, yi)
                losses = np.append(losses, loss)
                self.backpropagation(yi, predictions, learning_rate)
            if epoch < 10 or epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {losses.mean()}")

    def forward_pass(self, X, y=None):
        input_layer_output = self.input_layer.forward(X)
        hidden_layers_output = self.hidden_layers.forward(input_layer_output)
        predictions = self.output_layer.forward(hidden_layers_output)
        loss = [self.calculate_loss(prediction, y) for prediction in predictions] if y is not None else None
        return predictions, loss
    

input_layer = InputLayer(n_inputs=2, activation='tanh')
hidden_layers = HiddenLayers(prev_layer=input_layer, height=3, depth=2, activation='tanh')
output_layer = OutputLayer(prev_layer=hidden_layers, activation='sigmoid')

model = NeuralNetwork(input_layer, hidden_layers, output_layer)


df = pd.read_csv('data/creatures.csv')
X, y = df[['height', 'color']].to_numpy(), df['species'].to_numpy()

model.train(X, y, 150)

df = df.sample(frac=1).reset_index(drop=True)
X_test, y_test = df[['height', 'color']].to_numpy(), df['species'].to_numpy()