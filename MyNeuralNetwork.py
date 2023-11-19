from matplotlib import pyplot as plt
import numpy as np
import random

class NeuralNet:
    def __init__(self, layers, epochs, learning_rate, momentum, activation_function, validation_split):
        self.L = len(layers)
        self.n = layers
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.activation_function = activation_function
        self.validation_split = validation_split

        self.xi = [np.zeros(n) for n in layers]
        self.theta = [np.random.randn(n) for n in layers]
        self.delta = [np.zeros(n) for n in layers]

        self.w = [np.zeros((layers[i], layers[i - 1])) if i > 0 else None for i in range(self.L)]

        self.d_w = [np.zeros_like(w) for w in self.w if w is not None]
        self.d_theta = [np.zeros_like(theta) for theta in self.theta]

        self.d_w_prev = [np.zeros_like(w) for w in self.w if w is not None]
        self.d_theta_prev = [np.zeros_like(theta) for theta in self.theta]

    def activation(self, h):
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-h))
        elif self.activation_function == 'relu':
            return np.maximum(0, h)
        elif self.activation_function == 'linear':
            return h
        elif self.activation_function == 'tanh':
            return np.tanh(h)
        else:
            raise ValueError("Unknown activation function")

    def feed_forward(self, x_in):
        self.xi[0] = x_in
        for l in range(1, self.L):
            h = np.dot(self.w[l], self.xi[l - 1]) - self.theta[l]
            self.xi[l] = self.activation(h)
        return self.xi[-1]

    def back_propagate(self, y_in, z):
        self.delta[-1] = (y_in - z) * self.activation(self.xi[-1]) * (1 - self.activation(self.xi[-1]))

        for l in range(self.L - 2, 0, -1):
            self.delta[l] = np.dot(self.w[l + 1].T, self.delta[l + 1]) * self.activation(self.xi[l]) * (1 - self.activation(self.xi[l]))

    def update_weights_thresholds(self):
        for l in range(1, self.L):
            self.d_w[l] = -self.learning_rate * np.outer(self.delta[l], self.xi[l - 1]) + self.momentum * self.d_w_prev[l]
            self.w[l] += self.d_w[l]
            self.d_w_prev[l] = self.d_w[l]

            self.d_theta[l] = -self.learning_rate * self.delta[l] + self.momentum * self.d_theta_prev[l]
            self.theta[l] += self.d_theta[l]
            self.d_theta_prev[l] = self.d_theta[l]

    def fit(self, X, y):
        split_idx = int(len(X) * (1 - self.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        print(f"Training set shape: {X_train.shape}")
        print(f"Validation set shape: {X_val.shape}")

        for epoch in range(self.epochs):
            combined = list(zip(X_train, y_train))
            random.shuffle(combined)
            X_train, y_train = zip(*combined)

            for x, target in zip(X_train, y_train):
                output = self.feed_forward(x)
                self.back_propagate(target, output)
                self.update_weights_thresholds()

    def predict(self, X):
        return np.array([self.feed_forward(x) for x in X])

# Example usage
layers = [3, 9, 5, 1]
nn = NeuralNet(layers, epochs=100, learning_rate=0.01, momentum=0.9, activation_function='sigmoid', validation_split=0.2)

# Train the network 
# nn.fit(X, y)

# Make predictions
# predictions = nn.predict(X_test)

# train_errors, val_errors = nn.loss_epochs()
# plt.plot(train_errors, label='Training Error')
# plt.plot(val_errors, label='Validation Error')
# plt.xlabel('Epoch')
# plt.ylabel('Error')
# plt.legend()
# plt.show()
