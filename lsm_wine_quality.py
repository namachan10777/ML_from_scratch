#!/usr/bin/python3

import numpy as np
import pandas as pd
import numpy.linalg as lin

df = pd.read_csv("./dataset/wine_quality/winequality-red.csv", sep=";")

df_train = df[:int(len(df)/3*2)]
df_test = df[int(len(df)/3*2):]

target = "quality"
features = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
y = df_train[target].to_numpy().astype(np.float64)
g = df_train[features].to_numpy()

n = len(features)
means = np.zeros(n)
y_mean = np.mean(y)
y -= y_mean

for i in range(n):
    means[i] = np.mean(g[:,i])
    g[:,i] = g[:,i] - means[i]

a = lin.inv(g.T @ g) @ g.T @ y

# evaluation
g = df_test[features].to_numpy()
for i in range(n):
    g[:,i] -= means[i]
estimate = g @ a + y_mean
real = df_test[target].to_numpy()

print(estimate)
print(np.mean(np.sqrt((real - estimate)**2)))
