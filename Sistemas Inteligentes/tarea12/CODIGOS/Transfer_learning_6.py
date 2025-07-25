# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 21:46:34 2021

@author: Jaime
"""
#==============================================================================
# El código presentado desarrolla y evalúa un modelo de red neuronal multicapa 
# (MLP) para un problema de clasificación multiclase generado artificialmente 
# mediante la función make_blobs, utilizando tres centros de datos distribuidos 
# en dos dimensiones con una dispersión controlada. Primero, se generan y 
# codifican las etiquetas de salida en formato one-hot, y se divide el conjunto 
# en datos de entrenamiento y prueba. Luego, se define un modelo secuencial con 
# dos capas ocultas densas de cinco neuronas con activación ReLU e 
# inicialización He (he_uniform), seguido de una capa de salida con tres 
# neuronas (una por clase) y activación softmax. El modelo se compila utilizando 
# la pérdida de entropía cruzada categórica y el optimizador SGD, y se entrena 
# durante 100 épocas con validación sobre el conjunto de prueba. Posteriormente, 
# se evalúa su rendimiento en ambos subconjuntos, mostrando las precisiones 
# obtenidas, y se visualizan las curvas de entrenamiento correspondientes a la 
# pérdida y la exactitud tanto en entrenamiento como en validación. Esta 
# visualización facilita la interpretación del comportamiento del modelo, 
# identificando posibles casos de sobreajuste o aprendizaje insuficiente, y 
# sirve como base para ajustar hiperparámetros o arquitecturas en futuras 
# iteraciones.
#==============================================================================

# Caragra librerías
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
	# Descomponer el conjunto de entrenamiento y de prueba
	n_train = 500
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, trainy, testX, testy
 
#  Definir y construir el modelo sobre el dataset de entrenamiento
def fit_model(trainX, trainy, testX, testy):
	# Definir el modelo
	model = Sequential()
	model.add(Dense(5, input_dim=2, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(5, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(3, activation='softmax'))
	# Compilar el modelo
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	# Construir el modelo
	history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=1)
	return model, history
 
# Resumir el rendimiento del modelo de ajuste
def summarize_model(model, history, trainX, trainy, testX, testy):
	# Evaluar el modelo
	_, train_acc = model.evaluate(trainX, trainy, verbose=1)
	_, test_acc = model.evaluate(testX, testy, verbose=1)
	print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
	# Dibujar la pérdida durante el entrenamiento
	pyplot.subplot(211)
	pyplot.title('Loss')
	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
	# la exactitud durante el entrenamiento
	pyplot.subplot(212)
	pyplot.title('Accuracy')
	pyplot.plot(history.history['accuracy'], label='train')
	pyplot.plot(history.history['val_accuracy'], label='test')
	pyplot.legend()
	pyplot.show()
 
# Preparar los datos
trainX, trainy, testX, testy = samples_for_seed(2)

# Construir el modelo sobre el dataset de entrenamiento
model, history = fit_model(trainX, trainy, testX, testy)

# Evaluar el comportamiento del modelo
summarize_model(model, history, trainX, trainy, testX, testy)
