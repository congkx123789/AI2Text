import numpy as np

class TinyNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01
        self.b1 = np.zeros((hidden_size, 1))
        self.W2 = np.random.randn(output_size, hidden_size) * 0.01
        self.b2 = np.zeros((output_size, 1))

    def forward(self, X):
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = np.maximum(0, Z1)  # ReLU
        Z2 = np.dot(self.W2, A1) + self.b2
        exp_scores = np.exp(Z2 - np.max(Z2))
        return exp_scores / exp_scores.sum(axis=0, keepdims=True)

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=0)
