import random
import numpy as np
import pandas as pd


class ValueNode:
    def __init__(self, data: float, children: tuple = (), operation: str = ''):
        self.data = data
        self.gradient = 0
        self.previous = set(children)
        self.operation = operation
        self._backward = lambda: None
    
    def ensure_other_node(self, other):
        return other if isinstance(other, ValueNode) else ValueNode(other)
        
    def __add__(self, other):
        other = self.ensure_other_node(other)
        result_node = ValueNode(self.data + other.data, (self, other), '+')
        def _backward():
            self.gradient += result_node.gradient
            other.gradient += result_node.gradient
        result_node._backward = _backward
        return result_node

    def __mul__(self, other):
        other = self.ensure_other_node(other)
        result_node = ValueNode(self.data * other.data, (self, other), '*')
        def _backward():
            self.gradient += other.data * result_node.gradient
            other.gradient += self.data * result_node.gradient
        result_node._backward = _backward
        return result_node

    def __pow__(self, other):
        result_node = ValueNode(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.gradient += (other * self.data ** (other - 1)) * result_node.gradient
        result_node._backward = _backward
        return result_node

    def log(self):
        assert self.data > 0, "Logarithm not defined for zero or negative values."
        result_node = ValueNode(np.log(self.data), (self,), 'log')
        def _backward():
            self.gradient += (1 / self.data) * result_node.gradient
        result_node._backward = _backward
        return result_node

    def relu(self):
        result_node = ValueNode(0 if self.data < 0 else self.data, (self,), 'relu')
        def _backward():
            self.gradient += (result_node.data > 0) * result_node.gradient
        result_node._backward = _backward
        return result_node

    def sigmoid(self):
        sigmoid_value = 1 / (1 + np.exp(-self.data))
        result_node = ValueNode(sigmoid_value, (self,), 'sigmoid')
        def _backward():
            self.gradient += result_node.data * (1 - result_node.data) * result_node.gradient
        result_node._backward = _backward
        return result_node

    def backward(self):
        visited, ordered_nodes = set(), []
        def order_nodes(node):
            if node not in visited:
                visited.add(node)
                for child in node.previous:
                    order_nodes(child)
                ordered_nodes.append(node)
        order_nodes(self)
        self.gradient = 1
        for node in reversed(ordered_nodes):
            node._backward()

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __repr__(self):
        return f"ValueNode(data={self.data}, gradient={self.gradient})"


def dot(A: list[ValueNode], B: list[ValueNode]) -> ValueNode:
    assert len(A) == len(B), "Dot product requires arrays of the same length"
    return sum(a * b for a, b in zip(A, B))

def mean(X: list[float]) -> float:
    return sum(X) / len(X)


class NeuralComponent:
    def zero_grad(self):
        for p in self.parameters():
            p.gradient = 0

    def parameters(self):
        return []


class Neuron(NeuralComponent):
    def __init__(self, input_size=2, activation='sigmoid'):
        self.w = [ValueNode(np.random.randn() * 0.01) for _ in range(input_size)]
        self.b = ValueNode(np.random.randn() * 0.01)
        self.activation = activation
        self.activation_functions = {
            'sigmoid': self.sigmoid, 
            'relu': self.relu, 
            'linear': self.linear
            }

    def sigmoid(self, x):
        return x.sigmoid()

    def relu(self, x):
        return x.relu()

    def linear(self, x):
        return x

    def parameters(self):
        return self.w + [self.b]

    def forward(self, X):
        pre_activation = dot(X, self.w) + self.b
        activation_function = self.activation_functions[self.activation]
        return activation_function(pre_activation)


class NeuronLayer(NeuralComponent):
    def __init__(self, input_size:int, output_size:int, activation:str, loss:str='binary_cross_entropy'):
        self.neurons = [Neuron(input_size, activation) for _ in range(output_size)]
    
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def forward(self, X: list[float]) -> list[float]:
        return [n.forward(X) for n in self.neurons]


class FeedForwardNetwork(NeuralComponent):
    def __init__(self, input_size:int, output_size:int):
        self.n_inputs = input_size
        self.n_outputs = output_size
        self.layers = [NeuronLayer(input_size, input_size, 'relu'), NeuronLayer(input_size, output_size, 'sigmoid')]

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def binary_cross_entropy_loss(self, prediction, y):
        return -y * (1 - prediction).log() - (1 - y) * (1 - prediction).log()

    def train(self, X, y, learning_rate=0.05, epochs=10):
        for epoch in range(epochs + 1):
            losses = []
            for Xi, yi in zip(X, y):
                prediction, loss = self.forward(Xi, yi)
                losses.append(loss)
                self.zero_grad()
                loss.backward()
                for p in self.parameters():
                    p.data -= learning_rate * p.gradient
            if epoch < 10 or epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {mean(losses).data}")

    def forward(self, X, y):
        for layer in self.layers:
            X = layer.forward(X)
        return X, self.binary_cross_entropy_loss(X[-1], y)


df = pd.read_csv('data/creatures.csv')

X_train = df[['height', 'color']].values.tolist()
y_train = df['species'].values.tolist()

model = FeedForwardNetwork(2, 1)
model.train(X_train, y_train, epochs=10)