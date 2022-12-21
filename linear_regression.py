#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas
import numpy as np
import matplotlib.pyplot as plot
from pandas.plotting import scatter_matrix
#import seaborn as sns
# In[2]:
data = pandas.read_csv("car_data.csv")
# drop Column ID
data = data.drop('ID', axis=1)
# shuffle All Rows
data = data.sample(frac=1)
# In[3]:
#choosing the four main features
# d = data.corr()
# scatter_matrix(data)
# plot.show()

X = data[["carwidth", "curbweight", "enginesize", "horsepower"]]
y = data[["price"]]
for (columnName, columnData) in data.iteritems():
    plot.scatter(data.loc[:,columnName],y)
    plot.xlabel(columnName)
    plot.show()
#sns.pairplot(data)
#normalization
X = (X - X.min()) / (X.max() - X.min())
#normalization
y = (y - y.min()) / (y.max() - y.min())
# In[4]:
#initialization for gradients descent variables.
alpha = 0.01
theta = np.zeros(5)
theta = theta.reshape((-1, 1))
iterations = 1000

# In[5]:
#split the data into train and test.
M = len(data.index)
x_train = X[0:int(M / 2)]
y_train = y[0:int(M / 2)]
x_test = X[int(M / 2):]
y_test = y[int(M / 2):]
# normalization x_train
#x_train = (x_train - x_train.min()) / (x_train.max() - x_train.min())

# In[5]:
# insert X0 column
x_train.insert(0, "X0", 1, True) 

# In[6]:
#converting into numpy 
x_train = x_train.to_numpy()
y_train = y_train.to_numpy()


# In[8]:
#cost function calculation
def cost(X_train, Y_train, Theta):
    y_pred = X_train.dot(Theta)
    errors = np.subtract(y_pred, Y_train)
    return 1 / (2 * int(M / 2)) * errors.T.dot(errors)


# In[9]:
#gradient_descent calculations
def gradient_descent(X_train, Y_train, Theta, Alpha, Iterations):
    Costs = np.zeros(Iterations)
    for index in range(Iterations):
        predictions = X_train.dot(Theta)
        errors = np.subtract(predictions, Y_train)
        Rule = (Alpha / int(M / 2)) * X_train.T.dot(errors)
        Theta = Theta - Rule
        Costs[index] = cost(X_train, Y_train, Theta)
    return Theta, Costs


# In[10]:
#running model
FinalTheta, costs = gradient_descent(x_train, y_train, theta, alpha, iterations)
# In[12]:

x = np.arange(1, iterations + 1)
plot.plot(x, costs, color='red')
plot.title('MSE Over Each Iterations')
plot.ylabel('Cost For Each Iterations')
plot.xlabel('Iterations')
plot.show()

# In[13]:
#testing phase
print("increment Number Of Iterations += 100 in another 10 Test Cases And alpha Not Changed")
for i in range(20):
    iterations = iterations + 100
    FinalTheta, costs = gradient_descent(x_train, y_train, theta, alpha, iterations)
    x = np.arange(1, iterations + 1)
    plot.plot(x, costs, color='green')
    plot.title('MSE Over Each Iterations')
    plot.ylabel('Cost For Each Iterations')
    plot.xlabel('Iterations')
    plot.show()

# In[14]:
#testing phase
print("increment alpha += 0.001 in another 20 Test Cases And Iterations Not Changed")
for i in range(20):
    alpha = alpha + 0.001
    FinalTheta, costs = gradient_descent(x_train, y_train, theta, alpha, iterations)
    x = np.arange(1, iterations + 1)
    plot.plot(x, costs, color='blue')
    plot.title('MSE Over Each Iterations')
    plot.ylabel('Cost For Each Iterations')
    plot.xlabel('Iterations')
    plot.show()
