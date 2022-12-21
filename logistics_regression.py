import numpy as np
import pandas


class MyLogisticRegression:
    def __init__(self, learning_rate, iterations):
        self.learning_rate = learning_rate
        self.iterations = iterations

    # Function for model training    
    def fit(self, X, Y):
        self.m, self.features = X.shape
        # already transposed theta [0,0,0,0]
        self.theta = np.zeros(self.features)
        self.theta0 = 0
        self.X = X
        self.Y = Y

        # gradient descent
        for i in range(self.iterations):
            self.update_theta()
            # print(self.theta)
        return self

    def update_theta(self):
        # a = sigmoid( H(theta) ) and H(theta) = theta.X+theta0
        A = sigmoid((np.dot(self.X, self.theta) + self.theta0))
        # calculate change of both theta and theta0       
        tmp = (A - self.Y)
        dtheta = np.dot(self.X.T, tmp) / self.m
        dtheta0 = np.sum(tmp) / self.m

        # update thetas    
        self.theta = self.theta -self.learning_rate * dtheta
        self.theta0 = self.theta0 - self.learning_rate * dtheta0
        return self

    def predict(self, X):
        Z = sigmoid((X.dot(self.theta) + self.theta0))
        Y = [1 if y > 0.5 else 0 for y in Z]
        return Y


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def measureAccuracy(y_pred, y_test):
    return np.sum(y_pred == y_test) / len(y_test)


def main():
    url = "customer_data.csv"
    # load data
    dataset = pandas.read_csv(url)
    # add a ones coloum to the data
    # dataset.insert(0,'X0',1)
    # normalize data
    minmax = lambda x: (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))
    dataset = dataset.apply(minmax)
    # suffle data
    d = dataset.sample(frac=1)
    # split the suffled data
    x = d.drop(columns=['purchased']).to_numpy().reshape((-1, 2))
    y = d['purchased'].to_numpy()
    x_train = x[0:320]
    y_train = y[0:320]
    x_test = x[320:]
    y_test = y[320:]

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

    model = MyLogisticRegression(learning_rate=0.09, iterations=1800)

    model.fit(x_train, y_train)

    Y_pred = model.predict(x_test)

    print("Accuracy :  ", measureAccuracy(Y_pred, y_test) * 100)


if __name__ == "__main__":
    main()
