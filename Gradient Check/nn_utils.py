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


