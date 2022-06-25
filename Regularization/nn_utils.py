import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time

"""
Error & Updating
"""


def binary_cross_entropy(a, y):
    return -((y * np.log(a)) + ((1 - y) * np.log(1 - a)))

def update_param(param,dparam,lr):
    pass

"""
Activation Function
"""


def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def ReLU(z):
    return np.where(z >= 0, z, 0)


def LeakyReLU(z: float):
    return np.where(z >= 0, z, 0.01 * z)


"""
Derivative of Activation Function wrp. Z
"""


def dReLU(z: float):
    return np.where(z >= 0, 1, 0)


def dLeakyReLU(z: float):
    return np.where(z >= 0, 1, 0.01)


def dsigmoid(z: float):
    a = sigmoid(z)
    return a*(1 - a)


def dTanh(z: float):
    a = tanh(z)
    return 1 - a ** 2


"""
For Bi-Deep L layer Classification
"""


def cut_off_threshold(A, thr):
    return np.where(A >= thr, 1, 0)


"""
From Coursera
"""


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()
    
    
def predict_dec(param, X):
    """
    Used for plotting decision boundary.
    
    Arguments:
    param -- python dictionary containing your param 
    X -- input data of size (m, K)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache = L_model_forward(X, param)
    predictions = (a3>0.5)
    return predictions