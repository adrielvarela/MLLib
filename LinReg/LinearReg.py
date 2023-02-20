import numpy as np

class LinearRegression:
    def __init__(self, lr = 0.001, n_iters = 1000):
        self.learningRate = lr
        self.iters = n_iters
        self.weights = None
        self.bias = None
        self.meanSquareErr = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iters):
            y_prediction = np.dot(X, self.weights) + self.bias

            # leaving two out
            # np.dot includes the summation 
            dw = (1/n_samples) * np.dot(X.T, (y_prediction - y))
            db = (1/n_samples) * np.sum(y_prediction-y)

            self.weights = self.weights - (self.learningRate * dw)
            self.bias = self.bias - (self.learningRate * db)
        
    
    def predict(self, X):
        y_prediction = np.dot(X, self.weights) + self.bias
        return y_prediction
    
    def mse(self, predictions, actual_values):
        m = np.sum(predictions - actual_values) ** 2
        self.meanSquareErr = m
        return self.meanSquareErr


