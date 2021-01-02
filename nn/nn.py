#!/usr/bin/python3
import numpy as np
import copy

def sigmoid(x):
    return 1/(1+np.exp(-x))

def activation(X, w, b):
    return sigmoid(np.dot(X,w)+b)

def loss(X, y, w, b):
    dif = y - activation(X, w, b)
    return np.sum(dif**2/(2*len(y)), keepdims=True)

def predict(X, w, b):
    return np.where(activation(X, w, b)<0.5, -1.0, 1.0)

def accuracy(X, y, w, b):
    pre = predict(X, w, b)
    return np.sum(np.where(pre==y,1,0))/len(y)

def update(X, y, w, b, eta):
    a = (activation(X,w,b)-y)*activation(X, w, b)*(1-activation(X,w,b))
    a = a.reshape(-1,1)
    w -= eta * 1/float(len(y))*np.sum(a*X,axis=0)
    b -= eta * 1/float(len(y))*np.sum(a)
    return w,b

def init(input_size, output_size, layers):
    s_i = input_size
    ws = []
    bs = [np.ones(input_size)/10]
    layers = layers.copy()
    layers.append(output_size)
    for layer in layers:
        ws.append(np.ones([s_i, layer])/10)
        bs.append(np.ones(layer)/10)
        s_i = layer
    return ws, bs
