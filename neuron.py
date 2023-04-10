import numpy as np
import pandas as pd

class Neuron:
    def __init__(self):
        self.w = np.random.randn(2) * 0.01
        self.b = np.random.randn() * 0.01

    def sigmoid_activation(self, n):
        return 1 / (1 + np.exp(-n))
    
    def sigmoid_derivative(self, n):
        return  n * (1 - n)
    
    def binary_cross_entropy_loss(self, prediction, y):
        return -y * np.log(prediction) - (1 - y) * np.log(1 - prediction)

    def backpropagation(self, X, y, prediction, learning_rate):
        bias_gradient = self.sigmoid_derivative(prediction) * (prediction - y)
        weight_gradients = np.array([X[0] * bias_gradient, X[1] * bias_gradient])
        self.w = np.array([
            self.w[0] + -learning_rate * weight_gradients[0],
            self.w[1] + -learning_rate * weight_gradients[1]
            ]) 
        self.b = -learning_rate * bias_gradient

    def train (self, X, y, epochs=10, learning_rate=0.05):
        for epoch in range(epochs + 1):
            losses = np.array([])
            for i, x in enumerate(X):
                prediction, loss = self.forward_pass(x, y[i])
                losses = np.append(losses, loss)
                self.backpropagation(x, y[i], prediction, learning_rate)
            if epoch < 10 or epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {losses.mean()}")

    def forward_pass(self, X, y=None):
        prediction = self.sigmoid_activation(np.dot(self.w, X) + self.b)
        loss = self.binary_cross_entropy_loss(prediction, y) if y is not None else None
        return prediction, loss


df = pd.read_csv('fluffy_or_spikey.csv').sample(frac=1).reset_index(drop=True)

X_train, y_train = df[['height', 'color']].to_numpy(), df['species'].to_numpy()

model = Neuron() 
model.train(X_train, y_train, epochs=100)