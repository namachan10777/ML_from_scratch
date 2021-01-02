import nn
import pandas as pd
import numpy as np

df = pd.read_csv("./dataset/iris/iris.data", header=None)
y = df.iloc[0:100,4].values
y = np.where(y=='Iris-setosa',-1,1)
X = df.iloc[0:100,[0,1,2,3]].values

X_train = np.empty((80,4))
X_test = np.empty((20,4))
y_train = np.empty(80)
y_test = np.empty(20)
X_train[:40],X_train[40:] = X[:40],X[50:90]
X_test[:10],X_test[10:] = X[40:50],X[90:100]
y_train[:40],y_train[40:] = y[:40],y[50:90]
y_test[:10],y_test[10:] = y[40:50],y[90:100]

print(X_train)
print(X_test)
print(y_train)
print(y_test)

w = np.ones(4)/10
b = np.ones(1)/10

for _ in range(100):
    w, b = nn.update(X_train, y_train, w, b, eta=0.1)
    print(f'acc {nn.accuracy(X_test, y_test, w, b)} loss {nn.loss(X_train, y_train, w, b)}')

print(w)
print(b)
