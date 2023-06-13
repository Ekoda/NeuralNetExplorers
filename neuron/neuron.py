import numpy as np
import pandas as pd


class Neuron:
    def __init__(self, n_inputs=2):
        self.w = np.random.randn(n_inputs) * 0.1
        self.b = np.random.randn() * 0.1

    def sigmoid_activation(self, n):
        return 1 / (1 + np.exp(-n))
    
    def sigmoid_derivative(self, n):
        return n * (1 - n)
    
    def binary_cross_entropy_loss(self, prediction, y):
        return -y * np.log(prediction) - (1 - y) * np.log(1 - prediction)

    def binary_cross_entropy_loss_derivative(self, prediction, y):
        return -y / prediction + (1 - y) / (1 - prediction)

    def backpropagation(self, X, y, prediction, learning_rate):
        bias_gradient = self.sigmoid_derivative(prediction) * self.binary_cross_entropy_loss_derivative(prediction, y)
        weight_gradients = np.array([x * bias_gradient for x in X])
        self.w -= learning_rate * weight_gradients
        self.b -= learning_rate * bias_gradient

    def train (self, X, y, epochs=10, learning_rate=0.05):
        for epoch in range(epochs + 1):
            losses = np.array([])
            for Xi, yi in zip(X, y):
                prediction, loss = self.forward_pass(Xi, yi)
                losses = np.append(losses, loss)
                self.backpropagation(Xi, yi, prediction, learning_rate)
            if epoch < 10 or epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {losses.mean()}")

    def forward_pass(self, X, y=None):
        prediction = self.sigmoid_activation(np.dot(self.w, X) + self.b)
        loss = self.binary_cross_entropy_loss(prediction, y) if y is not None else None
        return prediction, loss


df = pd.read_csv('data/creatures.csv').sample(frac=1).reset_index(drop=True)

X_train, y_train = df[['height', 'color']].to_numpy(), df['species'].to_numpy()

model = Neuron()
model.train(X_train, y_train, epochs=100)
