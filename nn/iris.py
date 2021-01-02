import nn
import pandas as pd
import numpy as np
df = pd.read_csv("./dataset/iris/iris.data", header=None).to_numpy()
x_train = np.zeros([90,4])
y_train = np.empty(90, dtype=object)
x_train[0:30,:] = df[0:30, :4]
y_train[0:30] = np.full(30, 0)
x_train[30:60,:] = df[50:80, :4]
y_train[30:60] = np.full(30, 1)
x_train[60:90,:] = df[100:130, :4]
y_train[60:90] = np.full(30, 2)

x_test = np.zeros([60,4])
y_test = np.empty(60, dtype=object)
x_test[0:20,:] = df[30:50, :4]
y_test[0:20] = np.full(20, 0)
x_test[20:40,:] = df[80:100, :4]
y_test[20:40] = np.full(20, 1)
x_test[40:60,:] = df[130:150, :4]
y_test[40:60] = np.full(20, 2)

train = []
test = []
for i in range(90):
    y = np.zeros(3)
    y[y_train[i]] = 1.0
    train.append((x_train[i,:], y))

for i in range(60):
    y = np.zeros(3)
    y[y_test[i]] = 1.0
    test.append((x_test[i,:], y))

layers = [2]
bs, ws = nn.train(3.0, train, layers, 1000)
for (x,y) in test:
    hs, _ = nn.run(ws, bs, x)
    print(y, hs[-1])
