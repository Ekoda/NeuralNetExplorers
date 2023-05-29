import numpy as np
import pandas as pd

class RecurrentNeuron:
    def __init__(self, n_inputs=2):
        self.w = np.random.randn(n_inputs) * 0.01
        self.b = np.random.randn() * 0.01
        self.hw = np.random.randn() * 0.01

    def tanh_activation(self, n):
        return np.tanh(n)

    def tanh_derivative(self, n):
        return 1 - np.tanh(n)**2

    def mse_loss(self, prediction, y):
        return np.mean((prediction - y)**2)

    def mse_loss_derivative(self, prediction, y):
        return 2 * (prediction - y)

    def backpropagation(self, X_seq, y_seq, predictions, learning_rate):
        for t in reversed(range(len(X_seq))):
            bias_gradient = self.tanh_derivative(predictions[t]) * self.mse_loss_derivative(predictions[t], y_seq[t])
            weight_gradients = X_seq[t] * bias_gradient
            hidden_weight_gradient = bias_gradient * (predictions[t-1] if t > 0 else 0)
            self.w -= learning_rate * weight_gradients
            self.b -= learning_rate * bias_gradient
            self.hw -= learning_rate * hidden_weight_gradient

    def train(self, X_seq, y_seq, epochs=10, learning_rate=0.05):
        for epoch in range(epochs + 1):
            epoch_losses = np.array([])
            for Xi, yi in zip(X_seq, y_seq):
                predictions, losses = self.forward_pass(Xi, yi)
                epoch_losses = np.append(epoch_losses, losses)
                self.backpropagation(Xi, yi, predictions, learning_rate)
            if epoch < 10 or epoch % 10 == 0:
                print(f"Epoch: {epoch}, Loss: {epoch_losses.mean()}")

    def forward_pass(self, X_seq, y_seq=None):
        predictions = np.zeros(len(X_seq))
        losses = np.zeros(len(X_seq))
        for t in range(len(X_seq)):
            predictions[t] = self.tanh_activation(np.dot(self.w, X_seq[t]) + self.hw * (predictions[t-1] if t > 0 else 0) + self.b)
            losses[t] = self.mse_loss(predictions[t], y_seq[t]) if y_seq is not None else None
        return predictions, losses

sequence_length = 5

model = RecurrentNeuron(n_inputs=1)
df = pd.read_csv('data/stock_prices.csv')

X_train = df.index.values.reshape(-1, 1)
y_train = df['price'].values.reshape(-1, 1) / 10

X_sequences = [X_train[i:i+sequence_length] for i in range(len(X_train) - sequence_length + 1)]
y_sequences = [y_train[i:i+sequence_length] for i in range(len(y_train) - sequence_length + 1)]

model.train(X_sequences, y_sequences, epochs=50, learning_rate=0.0005)
