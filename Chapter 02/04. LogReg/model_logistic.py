import numpy as np
from sklearn.datasets import make_blobs


class LogisticRegression():
    def __init__(self, X, learning_rate, num_iters):
        self.lr = learning_rate
        self.num_iters = num_iters

     # m for training examples, n for features
        self.m, self.n = X.shape

    def sigmoid(self, z):
        
        """
          Activation function used to map any real value between 0 and 1.
          sigmoid(h(x)) = 1 / (1 + e^-((w.T * x) + b))
        """
        return 1 / (1 + np.exp(-z))


    def train(self, X, y):
        self.weights = np.zeros((self.n, 1))
        self.bias = 0

        for i in range(self.num_iters+1):
            # calculate hypothesis
            y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)

            """
               cost function formula:
               C(w, b) = -1/m * sum(i = 0 to m)(y*log h(x) + (1 - y)log(1 - h(x))         
            """

            # calculate cost
            cost = -1/self.m * np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))

            """
               delta C (w, b)/delta w = (h(x) - y) x(j to i)
               delta C (w, b)/ delta b = (h(x) - y)
            """

            # calculate weights and bias
            dw = 1/self.m * np.dot(X.T, (y_predict - y))
            db = 1/self.m * np.sum(y_predict - y)

            
            # Updating the parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db


            if i % 100 == 0:
                print(f'Cost after iteration {i}: {cost}')
              
        return self.weights, self.bias

    
    def predict(self, X):
        y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)
    
        pred_class = []
        #  y_predict > 0.5 = i > 0.5

        pred_class = [1 if i > 0.5 else 0 for i in y_predict]
        
        return np.array(pred_class)

    

    def accuracy(self, y, y_hat):
        accuracy = np.sum(y == y_hat) / len(y)

        return accuracy


    

if __name__ == '__main__':
    np.random.seed(1)
    X, y = make_blobs(n_samples=1000, centers=2)
    y = y[:, np.newaxis]


    logreg = LogisticRegression(X, 0.1, 1000)
    w, b = logreg.train(X, y)
    y_predict = logreg.predict(X)
    acc = logreg.accuracy(y, y_predict)
    

    print(f'Accuracy: {acc}')