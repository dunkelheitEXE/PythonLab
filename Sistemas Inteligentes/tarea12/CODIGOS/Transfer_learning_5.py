# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 19:13:45 2021

@author: Jaime
"""
#==============================================================================
# El código presentado combina dos fases complementarias de un flujo de trabajo 
# clásico en aprendizaje automático supervisado: visualización de datos 
# sintéticos y entrenamiento de un modelo de red neuronal multicapa (MLP). En 
# la primera parte, se generan múltiples conjuntos de datos bidimensionales 
# tridimensionales mediante la función make_blobs, que permite crear clústeres 
# artificiales con distribución gaussiana. Estos conjuntos de datos se 
# visualizan mediante diagramas de dispersión en diferentes subtramas, donde 
# cada punto se colorea según su clase, permitiendo observar la separabilidad 
# de los grupos generados. En la segunda parte del código, se entrena una red 
# neuronal sobre uno de los conjuntos previamente generados (con semilla fija), 
# utilizando dos capas ocultas con cinco neuronas cada una y funciones de 
# activación ReLU, junto con una capa de salida con softmax para clasificación 
# multiclase. El modelo se compila con la función de pérdida de entropía 
# cruzada categórica y se entrena durante 500 épocas con validación simultánea. 
# Tras el entrenamiento, se evalúa la precisión en los subconjuntos de 
# entrenamiento y prueba, y se grafican las curvas de pérdida y exactitud. 
# Finalmente, se guarda el modelo entrenado en el archivo 'model.h5' para su 
# reutilización, por ejemplo, en esquemas de transferencia de aprendizaje. Esta 
# práctica ilustra de forma clara cómo crear datasets sintéticos, entrenar 
# modelos sobre ellos y evaluar visualmente el rendimiento del proceso de 
# aprendizaje.
#==============================================================================

# Cargar paquetes
from sklearn.datasets import make_blobs
from numpy import where
from matplotlib import pyplot

# Generar un conjunto de datos de clasificación 2d 
X, y = make_blobs(n_samples=100, centers=3, n_features=3, cluster_std=0, random_state=1)

#####################################
# Generar muestras para problemas de blobs con una semilla aleatoria dada 
def samples_for_seed(seed):
	# Generar muestras
	X, y = make_blobs(n_samples=1000, centers=3, n_features=3, cluster_std=2, random_state=seed)
	return X, y
 
# Crear un diagrama de dispersión de puntos coloreados por valor de clase 
def plot_samples(X, y, classes=3):
	# Trazar puntos para cada clase 
	for i in range(classes):
		# Seleccionar índices de puntos con cada etiqueta de clase 
		samples_ix = where(y == i)
		# Trazar puntos para esta clase con un color dado 
		pyplot.scatter(X[samples_ix, 0], X[samples_ix, 1])
 
# Generar múltiples problemas 
n_problems = 2
for i in range(1, n_problems+1):
	# Especificar subtrama
	pyplot.subplot(210 + i)
	# Generar muestras 
	X, y = samples_for_seed(i)
	# Diagrama de dispersión de muestras 
	plot_samples(X, y)
# Figura de la trama 
pyplot.show()

################################

# Ajuste el modelo mlp en el problema 1 y guardar el modelo en un archivo 
from sklearn.datasets import make_blobs
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from matplotlib import pyplot
 
# Preparar ejemplos de blobs con una semilla aleatoria determinada 
def samples_for_seed(seed):
	# Generar muestras
	X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=seed)
	# Variable one hot encode de salida
	y = to_categorical(y)
	# Dividir en conjunto de entrenamiento y de prueba
	n_train = 500
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, trainy, testX, testy
 
# Construir el modelo sobre el conjunto de entrenamiento
def fit_model(trainX, trainy, testX, testy):
	# Definir modelo
	model = Sequential()
	model.add(Dense(5, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(3, activation='softmax'))
	# Compilar modelo
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	# Construir modelo
	history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=500, verbose=1)
	return model, history
 
# Resumen del nivel de ejecución del modelo construido
def summarize_model(model, history, trainX, trainy, testX, testy):
	# Evaluar el modelo
	_, train_acc = model.evaluate(trainX, trainy, verbose=1)
	_, test_acc = model.evaluate(testX, testy, verbose=0)
	print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
	# Dibujar la función de pérdida durante el entrenamiento
	pyplot.subplot(211)
	pyplot.title('Loss')
	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
	# Dibujar el nivel de precisión durante el entenamiento.
	pyplot.subplot(212)
	pyplot.title('Accuracy')
	pyplot.plot(history.history['accuracy'], label='train')
	pyplot.plot(history.history['val_accuracy'], label='test')
	pyplot.legend()
	pyplot.show()
 
# Preparar datos
trainX, trainy, testX, testy = samples_for_seed(1)

# Construir el modelo sobre el dataset
model, history = fit_model(trainX, trainy, testX, testy)

# Evaluar el comportamiento del modelo
summarize_model(model, history, trainX, trainy, testX, testy)

# Salvar el modelo en archivo
model.save('model.h5')