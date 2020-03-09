import numpy as np


class KNN:
    def __init__(self, k):
        # KNN state here
        # Feel free to add methods
        self.k = k

    def distance(self, featureA, featureB):
        diffs = (featureA - featureB) ** 2
        return np.sqrt(diffs.sum())

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    # training logic here
    # input is an array of features and labels

    def get_neighbors(self, Xtest):
        distance_list = []

        neighbors = []
        for t in self.X_train:
            d = self.distance(t, Xtest)
            distance_list.append((d, t))

        def key1(x):
            return x[1]

        list = sorted(distance_list, key=key1)

        for i in range(self.k):
            neighbors.append(list[i][1])
        return neighbors

    def predict(self, X):
        k_neighbors = self.get_neighbors(X)

        print(k_neighbors)
        output = [row[-1] for row in k_neighbors]
        prediction = max(set(output), key=output.count)
    # Run model here
    # Return array of predictions where there is one prediction for each set of features
        return prediction


class ID3:
    def __init__(self, nbins, data_range):
        # Decision tree state here
        # Feel free to add methods
        self.bin_size = nbins
        self.range = data_range

    def preprocess(self, data):
        # Our dataset only has continuous data
        norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
        categorical_data = np.floor(self.bin_size * norm_data).astype(int)
        return categorical_data

    def train(self, X, y):
        # training logic here
        # input is array of features and labels
        categorical_data = self.preprocess(X)

    def predict(self, X):
        # Run model here
        # Return array of predictions where there is one prediction for each set of features
        categorical_data = self.preprocess(X)
        return None


class Perceptron:
    def __init__(self, w, b, lr):
        # Perceptron state here, input initial weight matrix
        # Feel free to add methods
        self.lr = lr
        self.w = w
        self.b = b

    def train(self, X, y, steps):
        # training logic here
        # input is array of features and labels
        # self.w=np.zeroes(1+X.shape[1])
        # self.errors=[]

        for x in range(steps):
            error = 0
            for xi, target in zip(X, y):
                update = self.lr * (target - self.predict(xi))
                self.w[1:] += update * xi
                self.w[0] += update
                error += int(update != 0.0)
            self.b.append(error)
        return self

    def net_input(self, X):
        return np.dot(X, self.w[1:] + self.w[0])

    def predict(self, X):
        # Run model here
        # Return array of predictions where there is one prediction for each set of features

        return np.where(self.net_input(X) >= 0.0, 1, -1)

