#!/usr/bin/python3

import numpy as np
import pandas as pd
import numpy.linalg as lin

df = pd.read_csv("./dataset/wine_quality/winequality-red.csv", sep=";")

# 前半2/3を訓練データ、後半1/3をテストデータに使う
df_train = df[:int(len(df)/3*2)]
df_test = df[int(len(df)/3*2):]

# qualityを予測する
target = "quality"
features = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]

y = df_train[target].to_numpy().astype(np.float64)
g = df_train[features].to_numpy()
# 切片分を挿入
g = np.insert(g, 0, 1, axis=1)
a = lin.inv(g.T @ g) @ g.T @ y

y_test = df_test[target].to_numpy().astype(np.float64)
g_test = df_test[features].to_numpy()
g_test = np.insert(g_test, 0, 1, axis=1)

# 予測してR^2を計算
r2 = 1 - np.sum(np.power(y_test - g_test @ a, 2)) / np.sum(np.power(y - np.mean(y), 2))
print(r2)
