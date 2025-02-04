import numpy as np

class LogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, batch_size=32, max_iters=1000):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iters = max_iters
        self.weights = None  # Model parameters

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def compute_loss(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def compute_gradient(self, X_batch, y_batch):
        N = X_batch.shape[0]
        y_pred = self.sigmoid(X_batch @ self.weights)
        return (1 / N) * X_batch.T @ (y_pred - y_batch)  # Gradient of cross-entropy loss

    def fit(self, X, y):
        N, D = X.shape
        self.weights = np.zeros(D)  # Initialize weights to zero

        for _ in range(self.max_iters):
            # Mini-batch selection
            indices = np.random.choice(N, self.batch_size, replace=False)
            X_batch, y_batch = X[indices], y[indices]

            # Compute gradient and update weights
            gradient = self.compute_gradient(X_batch, y_batch)
            self.weights -= self.learning_rate * gradient

    def predict_proba(self, X):
        return self.sigmoid(X @ self.weights)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
