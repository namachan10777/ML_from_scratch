#!/usr/bin/python3
import numpy as np
import copy

def sigmoid(x):
    return 1/(1+np.exp(-x))

def activation(X, w, b):
    return sigmoid(np.dot(X,w)+b)

def sigmoid_d(x):
    return (1-sigmoid(x))*sigmoid(x)

def loss(x):
    return np.sum(np.power(y - activation(X, w, b))/(2*len(y)), keepdims=True)
