# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 21:01:08 2021

@author: Jaime
"""

# Transferir el aprendizaje con el modelo mlp en el problema 2 
from sklearn.datasets import make_blobs
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.models import load_model
from matplotlib import pyplot
 
# Preparar ejemplos de blobs con una semilla aleatoria determinada 
def samples_for_seed(seed):
	# Generar muestras
	X, y = make_blobs(n_samples=1000, centers=3, n_features=2, cluster_std=2, random_state=seed)
	# Variable de salida de codificación one hot 
	y = to_categorical(y)
	# Dividir en conjunto de entrenamiento y de prueba
	n_train = 500
	trainX, testX = X[:n_train, :], X[n_train:, :]
	trainy, testy = y[:n_train], y[n_train:]
	return trainX, trainy, testX, testy
 
# Cargar y reajustar el modelo en un conjunto de datos de entrenamiento 
def fit_model(trainX, trainy, testX, testy):
	# Almacenar el modelo
	model = load_model('model.h5')
	# Compilar el modelo
	model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	# Re-ajustar el modelo
	history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=100, verbose=1)
	return model, history
 
# Resumir el rendimiento del modelo de ajuste 
def summarize_model(model, history, trainX, trainy, testX, testy):
	# Evaluar el modelo
	_, train_acc = model.evaluate(trainX, trainy, verbose=1)
	_, test_acc = model.evaluate(testX, testy, verbose=1)
	print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
	# Plotear pérdida durante el entrenamiento 
	pyplot.subplot(211)
	pyplot.title('Loss')
	pyplot.plot(history.history['loss'], label='train')
	pyplot.plot(history.history['val_loss'], label='test')
	pyplot.legend()
	# Plotear exactitud durante el entrenamiento
	pyplot.subplot(212)
	pyplot.title('Accuracy')
	pyplot.plot(history.history['accuracy'], label='train')
	pyplot.plot(history.history['val_accuracy'], label='test')
	pyplot.legend()
	pyplot.show()
 
# Preparar datos
trainX, trainy, testX, testy = samples_for_seed(2)
# Construir el modelo sobre el cojnuto de entrenamiento
model, history = fit_model(trainX, trainy, testX, testy)
# Evaluar el comportamiento del modelo
summarize_model(model, history, trainX, trainy, testX, testy)

