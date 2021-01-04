import pandas as pd
import numpy as np
import copy

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

def sigmoid(x):
    return 1/(1+np.exp(-x))

w0 = np.ones([4,2])/10
b0 = np.ones([1,2])/10
w1 = np.ones([2,1])/10
b0 = np.ones([1,1])


# y = x
def g_out(x):
    return x

def g_out_d(x):
    return np.ones_like(x)

def err(y, x):
    return 1/(2*float(len(y))) * np.sum(np.power(x - y, 2), axis=0)

def err_d(y, x):
    return x - y

def g(x):
    return 1/(1+np.exp(-x))

def g_d(x):
    return g(x)*(1-g(x))

def next(x, w, b):
    return np.add(np.dot(x, w.T), b)

def delta_out(y, out, h_out):
    return err_d(y, out) * g_out_d(h_out)

def delta(delta_next, w, h):
    return np.dot(delta_next, w) * g_d(h)

def run(X, ws, bs):
    h = X
    x = X
    xs = [h]
    hs = [x]
    for n in range(len(ws)-1):
        h = next(x, ws[n], bs[n])
        x = g(h)
        hs.append(h)
        xs.append(x)
    h = next(x, ws[-1], bs[-1])
    x = g_out(h)
    hs.append(h)
    xs.append(x)
    return hs, xs

def update(X, y, ws, bs, eta):
    y = y.reshape(-1,1)
    hs, xs = run(X, ws, bs)
    d = delta_out(y, xs[-1], hs[-1])
    for n in reversed(range(len(ws))):
        w_ = np.copy(ws[n])
        b_ = np.copy(bs[n])
        J, I = ws[n].shape
        for j in range(J):
            b_[j] -= eta * np.sum(d[:,j]) / float(len(y))
            for i in range(I):
                w_[j,i] -= eta * np.dot(d[:,j],xs[n][:,i]) / float(len(y))
        d = delta(d, ws[n], hs[n])
        ws[n] = w_
        bs[n] = b_
    e = err(y, hs[-1])
    return (ws, bs, e)

w0 = np.array([[0.0, 0.0]])
b0 = np.array([1.0])
ws = [w0]
bs = [b0]

X_train = np.empty([25,2])
y_train = np.empty(25)
for i in range(5):
    for j in range(5):
        X_train[i*5+j,0] = i
        X_train[i*5+j,1] = j
        y_train[i*5+j] = i+j
print(X_train)
print(y_train)

for _ in range(100):
    ws, bs, e = update(X_train, y_train, ws, bs, 0.1)
    print(f'err {e}')

print(ws)
print(bs)
