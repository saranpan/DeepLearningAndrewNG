#!/usr/bin/env python
# coding: utf-8

# status : Add Weight Init to Deep - L - Layer model
# 
# follow https://github.com/Kulbear/deep-learning-coursera/blob/master/Improving%20Deep%20Neural%20Networks%20Hyperparameter%20tuning%2C%20Regularization%20and%20Optimization/Initialization.ipynb

# In[134]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time


# In[135]:


"""
Error & Updating
"""


def binary_cross_entropy(a, y):
    return -((y * np.log(a)) + ((1 - y) * np.log(1 - a)))


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


def dTanh(z: float):
    a = tanh(z)
    return 1 - a ** 2


"""
For Bi-Deep L layer
"""


def thresholder(A, thr):
    return np.where(A >= thr, 1, 0)


# In[136]:


"""
Initiate parameter
"""


def initiate_param(hyperparam: list,initialization = 'random',seed:int=42):
    """
    Initiate the paramaters W, B for each layer
    input : hyperparameter dictionary (specifically, we need the number of unit for every layer)
    """
    if seed:
        np.random.seed(seed)
    
    n_unit = hyperparam["n_unit"]

    if initialization == 'zero':
        param = initialization_zero(n_unit)
    elif initialization == 'random':
        param = initialization_random(n_unit)
    elif initialization == 'He':
        param = initialization_he(n_unit)
    elif initialization == 'Xavier':    
        param = initialization_xavier(n_unit)
    else: #default : random
        print(f'''There is no weight initialization called "{initialization}"
              switch to default initialization random
              ''')
        param = initialization_random(n_unit)
        
    return param


# In[137]:


def initialization_zero(n_unit:list):
    """
    Initialize both weight and bias as zeros
    
    Argument:
    n_units : number of units for every layer respectively
    
    Returns:
    param : parameter W, B for every layer from 1 to L
    """
    
    L = len(n_unit) - 1  #Exclude input layer to calculating L
    param = {}
    
    for l in range(1,L):
        param["W" + str(l)] = np.zeros(shape=(n_unit[l], n_unit[l-1])) * 0.01 # Uniform(0,1] * 0.01
        param["b" + str(l)] = np.zeros(shape=(n_unit[l], 1))
    return param
    


# In[138]:


def initialization_random(n_unit:list,scale:int=0.01):
    """
    Initialize weight randomly with Normal(mean=0,sigma=1)
    Initialize bias as zeros
    
    Argument:
    n_units : number of units for every layer respectively
    
    Returns:
    param : parameter W, B for every layer from 1 to L
    """
    L = len(n_unit) - 1  #Exclude input layer to calculating L
    param = {}
    
    """
    scale : variance of the random variable
    y = scale * x
    var(y) = var(scale*x)
    var(y) = scale^2 * x
    """
    for l in range(1,L):
        param["W" + str(l)] = np.random.randn(n_unit[l], n_unit[l-1]) * scale # Normal(0,1) * scale 
        param["b" + str(l)] = np.zeros(shape=(n_unit[l], 1))
    return param
    


# In[139]:


def initialization_he(n_unit:list):
    """
    Initialize weight randomly with Normal(mean=0,sigma=(2/fan_in))
    Initialize bias as zeros
    
    Argument:
    n_units : number of units for every layer respectively
    
    Returns:
    param : parameter W, B for every layer from 1 to L
    """
    
    L = len(n_unit) - 1  #Exclude input layer to calculating L
    param = {}
    
    for l in range(1,L):
        fan_in = n_unit[l-1]
        
        param["W" + str(l + 1)] = np.random.randn(n_unit[l], n_unit[l-1]) * np.sqrt(2/fan_in) 
        param["b" + str(l + 1)] =  np.zeros(shape=(n_unit[l], 1))
    return param
    


# In[140]:


def initialization_xavier(n_unit:list):
    """
    Initialize weight randomly with Normal(mean=0,sigma=(1/fan_avg))
    Initialize bias as zeros
    
    Argument:
    n_units : number of units for every layer respectively
    
    Returns:
    param : parameter W, B for every layer from 1 to L
    """
    
    L = len(n_unit) - 1  #Exclude input layer to calculating L
    param = {}
    
    for l in range(1,L):
        fan_in , fan_out = n_unit[l-1] , n_unit[l]
        fan_avg = 1/2 * (fan_in + fan_out)
        
        param["W" + str(l + 1)] = np.random.randn(n_unit[l], n_unit[l-1]) * np.sqrt(1/fan_avg) 
        param["b" + str(l + 1)] =  np.zeros(shape=(n_unit[l], 1))
    return param
    


# In[141]:


"""
Forward Propagation Unit
"""


def linear_forward(A_prev, W, b):
    Z = np.dot(W, A_prev) + b
    linear_cache = (
        A_prev,
        W,
        b,
    )  # A :for dZ, W for dA & to get updating, b for updating , dA for dZ
    return Z, linear_cache


def linear_activation_forward(A_prev, W, b, activation_function):
    Z, linear_cache = linear_forward(A_prev, W, b)

    if activation_function == "sigmoid":
        A = sigmoid(Z)
    elif activation_function == "tanh":
        A = tanh(Z)
    elif activation_function == "ReLU":
        A = ReLU(Z)
    elif activation_function == "LeakyReLU":
        A = LeakyReLU(Z)

    cache = (Z, linear_cache)  # (Z,A_prev,W,b)

    return A, cache


def L_model_forward(X, param, activation_function="ReLU"):
    """
    Forward propagation unit from input to output layer
    Argument
    1. X --- Input denoted as A[0]
    2. param --- Weight and Bias of every layer
    3. activation_function --- the activation function of hidden layers (default:ReLU)

    Return
    1. A --- Output A[L] from the propagation (Z[L] with sigmoid activation function)
    2. caches --- the cache of every layer l ; [Z[l] , A[l-1], W[l], b[l]]
              --- L elements , each elements (array forms with 4 sub-arrays) contain the cache of its layer
                  linear_cache : Z[l]
                  activation_cache : A[l-1], W[l], b[l]
    """

    A = X
    L = (
        len(param) // 2
    )  # param stores the weight and bias for L layer, hence len(param) = 2L

    caches = []

    for l in range(L):  # l = 0,1,2,..,L-1
        A_prev = A
        W = param["W" + str(l + 1)]
        b = param["b" + str(l + 1)]

        A, cache = linear_activation_forward(A_prev, W, b, activation_function)
        caches.append(cache)  # append cache at layer l+1

    return A, caches


# In[142]:


"""
Backward Propagation Unit
"""


def linear_backward(dZ, cache):
    """
    Use dZ from the layer l to obtain dW,dB,dA_prev
    Arguments:
      dZ -- Gradient of the cost with respect to the linear output (of current layer l)
      cache -- tuple of values (Z,(A_prev, W, b)) coming from the forward propagation in the current layer (We use only linear cache anyway)

    Returns:
      dA_prev --- Gradient of the cost with respect to the activation node at the previous layer
      dW --- Gradient of the cost with the weight in this layer
      db --- Gradient of the cost with the bias in this layer
    """
    _, activation_cache = cache  # We use only activation cache
    (a_prev, W, _) = activation_cache  # We do not use b to update
    m = dZ.shape[1]

    dW = (1 / m) * np.dot(dZ, a_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation_function):
    """
    Input dA to find dZ, then use dZ to obtain dW,dB,dA_prev
    """
    if activation_function == "ReLU":
        g_ = dReLU
    elif activation_function == "LeakyReLU":
        g_ = dLeakyReLU
    elif activation_function == "tanh":
        g_ = dTanh
    else:
        print(
            f"The activation function {activation_function} not found, ReLU as default"
        )
        g_ = dReLU

    linear_cache, _ = cache  # We use only linear output cache
    Z = linear_cache

    dZ = dA * g_(Z)
    dA_prev, dW, db = linear_backward(dZ, cache)

    return dA_prev, dW, db


def L_model_backward(A, Y, cache, activation_function="ReLU"):
    """
    Do the whole backward propagation model by retrieving the output from forward propagation model and the actual output
    Arguments:
    A --- A at the layer L
    y --- an actual output
    cache --- cache from the forward propagation
    activation_function --- activation function for the hidden layer
    Return:
     grads  -- A dictionary with the gradients
               grads["dA" + str(l)] = ...
               grads["dW" + str(l)] = ...
               grads["db" + str(l)] = ...
    """
    L = len(cache)  # cache for each layer
    grads = {}
    dA = np.divide(1 - Y, 1 - A) - np.divide(
        Y, A
    )  # dA_[L] : Input for the first linear activation backward

    for l in reversed(range(L)):

        current_cache = cache[l]
        (linear_cache, activation_cache) = current_cache
        Z = linear_cache
        a_prev, W, b = activation_cache  # Start with Z_[L] , A_[L-1], W_[L], b_[L]

        dA_prev, dW, db = linear_activation_backward(
            dA, current_cache, activation_function
        )

        grads["dW" + str(l + 1)] = dW
        grads["db" + str(l + 1)] = db

        dA = dA_prev

    return grads


# In[143]:


def update_param(param, grads, lr=1e-4):
    """
    Argument:
    1. param -- The current parameter (W1,W2,...,WL,b1,b2,...bL)
    2. grads -- the dictionary of gradient that was obtained from L_model_backward function
    3. lr (default=1e-4) : Learning rate
    Returns:
    1. updated_param -- The parameter that got updated
    """

    L = len(param) // 2  # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        param["W" + str(l + 1)] = (
            param["W" + str(l + 1)] - lr * grads["dW" + str(l + 1)]
        )
        param["b" + str(l + 1)] = (
            param["b" + str(l + 1)] - lr * grads["db" + str(l + 1)]
        )

    return param


# In[144]:


def cost_reporter(A, Y, m, epoch, Epoch):
    """
    report the cost for each epoch
    cost function : Binary cross entropy
    Arguments:
    A --- predicted value from L-Forward model
    y --- actual output
    m --- total observations
    epoch --- current epoch
    Epoch --- total epoch
    """
    loss = binary_cross_entropy(A, Y)
    cost = np.divide(
        loss, m
    )  # No significant difference in speed when compare to '/' though
    cost = np.sum(cost, axis=1)
    print(f"Epoch {epoch}/{Epoch} : ===Cost=== : {np.squeeze(cost)}")


# In[145]:


class Binary_Deep_L_Layer:
    """
    A Deep neural network with L layers
    - Able to fit with the predictors (X) and the response (Y)
    - Able to predict_proba and predict with threshold
    To see the last fit model parameter, uses self.param where self refer to the fit model
    """

    def __init__(self, hyperparam: dict):
        """
        Launch the Deep_L_layer with the given hyperparameter
        Arguments:
        hyperparam: A dictionary with key:
         L --- Number of Layers (Hidden layer(s) + Output layer)
         n_unit --- Number of units of that L layer
         lr --- Learning rate
         forward_activation_function --- Activation function for all hidden layer(s) in forward model (ReLU,LeakyReLU,tanh,sigmoid)
         backward_activation_function --- Activation function for all hidden layer(s) in backward model (ReLU,LeakyReLU,tanh,sigmoid)
          {"L" : 5,
          "n_unit" : [nrow,8,6,4,2,1],
          "lr" : 1e-5,
          "forward_activation_function" : 'tanh' ,
          "backward_activation_function" : 'ReLU' }
        Supported activation
        """
        self.hyperparam = hyperparam  # assume include nrow in dict

        # Explicit hyperparameter attributes
        self.L = hyperparam["L"]
        self.lr = hyperparam["lr"]
        self.forward_activation_function = hyperparam["forward_activation_function"]
        self.backward_activation_function = hyperparam["backward_activation_function"]
        
    def compiles(self, initialization:str='random', optimizer='adam', loss='binary_cross_entropy'):
        """
        Develop soon (optional hyperparameter)
        """
        self.initialization = initialization
        self.optimizer = optimizer
        self.loss = loss
        
    def fit(
        self,
        X: pd.DataFrame,
        Y: pd.Series,
        Epochs: int = 1000,
        verbose: bool = True,
        new_fit: bool = True,
    ):
        """
        Fit the launched Deep L layer with the given data X , Y

        Arguments:
         X --- Pandas Dataframe of predictors
         Y --- Pandas Series of response (0 : negative, 1:positive)
         Epoch --- number of epochs (default : 1000)
         verbose --- report the epochs every 1000 epoch
         new_fit --- reset parameters and generate parameters
        """

        ## First, we initiate the attributes

        # We turn Dataframe into Numpy format
        X = X.to_numpy().T
        Y = Y.to_numpy().T
        nrow = np.shape(X)[0]

        # Assign class attribute
        self.X = X
        self.Y = Y
        self.m = Y.shape[1]
        self.Epochs = Epochs

        ## Second, we fit
        if new_fit:
            self.param = initiate_param(hyperparam = self.hyperparam,
                                        initialization = self.initialization)

        for epoch in range(self.Epochs):
            A, cache = L_model_forward(
                self.X, self.param, activation_function=self.forward_activation_function
            )

            if verbose and epoch % 1000 == 0:
                cost_reporter(A, self.Y, self.m, epoch, self.Epochs)

            grads = L_model_backward(
                A, self.Y, cache, self.backward_activation_function
            )
            self.param = update_param(self.param, grads, lr=self.lr)

    def predict_proba(self, X: pd.DataFrame):
        """
        Predict probability of the observation given input X

        Arguments:
         X --- Pandas Dataframe or Series of predictors
        """
        X = X.to_numpy().T

        A_prob, _ = L_model_forward(
            X, self.param, activation_function=self.forward_activation_function
        )

        return A_prob

    def predict(self, X, threshold: float = 0.5):
        """
        Predict the observation given input X

        Arguments:
         X --- Pandas Dataframe or Series of predictors
        """
        A_prob = self.predict_proba(X)
        A_pred = thresholder(A_prob, threshold)
        return A_pred

    def __repr__(self):
        return f"Deep_L_Layer({self.hyperparam})"

    def __str__(self):
        return f"A Deep {self.L} Neural network with learning rate = {self.lr} (Forward activation :{self.forward_activation_function},Backward activation :{self.backward_activation_function})"


# 
# 
# ---
# 
# 

# Test case

# In[146]:


df = pd.read_csv('Dataset/churn_small3.csv')

hyperparam = {"L" : 5,
              "n_unit" : [3,8,6,4,2,1],
              "lr" : 1e-5,
              "forward_activation_function" : 'ReLU',
              "backward_activation_function" : 'ReLU',
              "keep_prob_sequence" : [1,0.5,0.6,0.7,1,1]}   #Dropout => No dropout = None OR [1,1,1,1,1,1]

X = df[['tenure','TotalCharges'	,'PaperlessBilling']]
Y = df[['Churn']]


# In[ ]:





# In[147]:


model = Binary_Deep_L_Layer(hyperparam)


# In[148]:


hyperparam = {"L" : 5,
              "n_unit" : [3,8,6,4,2,1],
              "lr" : 1e-5,
              "forward_activation_function" : 'ReLU',
              "backward_activation_function" : 'ReLU',
              "keep_prob_sequence" : [1,0.5,0.6,0.7,1,1]}


# In[150]:


initiate_param(hyperparam,initialization = 'zero',seed=42)

