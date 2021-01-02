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

ws, bs = nn.init(4, 1, [3,3])

print(ws)
print(bs)
print(nn.sigmoid(np.add(np.dot(X_train, ws[0]), bs[1])))
