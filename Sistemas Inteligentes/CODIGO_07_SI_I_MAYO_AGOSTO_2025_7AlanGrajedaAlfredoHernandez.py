import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

dataset = pd.read_csv('modern_teen_mental_health_main.csv')
print(dataset.columns)

target = dataset["stress_level"]
binario = dataset[target.isin([1,8])]

X = binario[['screen_time_hours', 'sleep_hours']].values
y = binario['stress_level'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


clf = Perceptron(max_iter=1000, eta0=0.1, random_state=0)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Precisión:", accuracy_score(y_test, y_pred))

# Pesos sinápticos (uno por cada característica)
print("Pesos sinápticos (coeficientes):", clf.coef_)

# Polarización (bias o término independiente)
print("Polarización (bias):", clf.intercept_)

from sklearn.metrics import accuracy_score

# Predecimos con el modelo entrenado
y_pred = clf.predict(X_test)

# Calculamos la precisión
precision = accuracy_score(y_test, y_pred)

print("Precisión del modelo:", precision)



