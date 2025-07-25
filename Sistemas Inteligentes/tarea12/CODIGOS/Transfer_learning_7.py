# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 23:31:56 2021

@author: Jaime
"""

#==============================================================================
# El código presentado implementa un experimento comparativo para evaluar el 
# rendimiento de un modelo de red neuronal multicapa (MLP) entrenado de forma 
# independiente frente a variantes entrenadas mediante aprendizaje por 
# transferencia en un contexto sintético. Para ello, primero se generan datos 
# artificiales en forma de grupos (blobs) tridimensionales mediante la función 
# make_blobs, y se codifican las salidas con one-hot encoding. Se construye un 
# modelo base MLP con dos capas ocultas densas con función de activación ReLU y 
# una capa de salida softmax para clasificación multiclase. El experimento 
# evalúa dos enfoques: (1) entrenamiento desde cero de modelos independientes y 
# (2) uso de un modelo previamente entrenado (model.h5) donde se fijan 
# (congelan) una o más capas iniciales para evaluar el impacto de la 
# transferencia de conocimiento en diferentes configuraciones. Ambos enfoques 
# se repiten 30 veces con diferentes inicializaciones para obtener una 
# distribución de puntuaciones de precisión sobre el conjunto de prueba. 
# Finalmente, los resultados de precisión se visualizan mediante diagramas de 
# caja (boxplot), lo que permite comparar estadísticamente la variabilidad y el 
# rendimiento medio de los modelos entrenados de forma independiente y los que 
# utilizan transferencia de aprendizaje con diferentes números de capas 
# congeladas. Este enfoque permite explorar empíricamente la estabilidad y 
# eficacia de la transferencia de parámetros en tareas similares, incluso en 
# problemas generados artificialmente.
#==============================================================================

# Compare el rendimiento del modelo mlp independiente con el aprendizaje de transferencia 
from sklearn.datasets import make_blobs
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.models import load_model
from matplotlib import pyplot
from numpy import mean
from numpy import std
  
# Preparar ejemplos de blobs con una semilla aleatoria determinada 
def samples_for_seed(seed):
	# Generar muestras
	X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=seed)
	# Una variable de salida de codificación one hot
	y = to_categorical(y)
	# Dividir en conjunto de entrenamiento y de prueba 
	n_train = 500
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, trainy, testX, testy
 
# Definir y ajustar el modelo en un conjunto de datos de entrenamiento 
def fit_model(trainX, trainy):
	# Definir el modelo
	model = Sequential()
	model.add(Dense(5, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(3, activation='softmax'))
	# Compilar el modelo
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	# Construir el modelo
	model.fit(trainX, trainy, epochs=100, verbose=1)
	return model
 
# Evaluación repetida de un modelo independiente 
def eval_standalone_model(trainX, trainy, testX, testy, n_repeats):
	scores = list()
	for _ in range(n_repeats):
		# Definir y ajustar un nuevo modelo en el conjunto de datos de entrenamiento
		model = fit_model(trainX, trainy)
		# Evaluar el modelo en el conjunto de datos de prueba 
		_, test_acc = model.evaluate(testX, testy, verbose=1)
		scores.append(test_acc)
	return scores
 
# Evaluación repetida de un modelo con transferencia de aprendizaje 
def eval_transfer_model(trainX, trainy, testX, testy, n_fixed, n_repeats):
	scores = list()
	for _ in range(n_repeats):
		# Almacenar el modelo
		model = load_model('model.h5')
		# Marcar los pesos de las capas como fijos o no entrenables 
		for i in range(n_fixed):
			model.layers[i].trainable = False
		# Volver a compilar modelo 
		model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
		# Ajustar el modelo en el conjunto de datos de entrenamiento 
		model.fit(trainX, trainy, epochs=100, verbose=1)
		# Evaluar el modelo en el conjunto de datos de prueba
		_, test_acc = model.evaluate(testX, testy, verbose=1)
		scores.append(test_acc)
	return scores
 
# Preparar datos para el problema 2
trainX, trainy, testX, testy = samples_for_seed(2)
n_repeats = 30
dists, dist_labels = list(), list()
 
# Evaluación repetida del modelo independiente
standalone_scores = eval_standalone_model(trainX, trainy, testX, testy, n_repeats)
print('Standalone %.3f (%.3f)' % (mean(standalone_scores), std(standalone_scores)))
dists.append(standalone_scores)
dist_labels.append('standalone')
 
# Evaluación repetida del modelo de aprendizaje por transferencia, variación de capas fijas 
n_fixed = 3
for i in range(n_fixed):
	scores = eval_transfer_model(trainX, trainy, testX, testy, i, n_repeats)
	print('Transfer (fixed=%d) %.3f (%.3f)' % (i, mean(scores), std(scores)))
	dists.append(scores)
	dist_labels.append('transfer f='+str(i))
 
# Diagrama de caja de las distribuciones de puntuación 
pyplot.boxplot(dists, labels=dist_labels)
pyplot.show()



