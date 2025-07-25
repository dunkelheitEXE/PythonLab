# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 22:35:57 2021

@author: Jaime
"""
# Ejemplo del uso de un modelo pre-entrenado como un clasificador
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Model
from pickle import dump

# Almacenar una imagen 
image = load_img('C:/Users/Asus.S510UNR/Desktop/MAYO-AGOSTO-2025/UPMH-MAYO-AGOSTO-2025/CODIGOS/CODIGOS-MAYO-AGOSTO-2025/SEMANA-4/dog.jpg', target_size=(224, 224))

# Convertir la imagen de pixeles en un arreglo de numpy
image = img_to_array(image)
print(image)

# Remodelar datos para el modelo 
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
print(image.shape[0])
print(image.shape[1])
print(image.shape[2])
print(image)

# Preparar la imagen para aplicarla al modelo VGG16
image = preprocess_input(image)
print(image)

# Guardar el modelo
model = VGG16()
model.summary()

# Estimar la probabilidad en todas las clases de la salida 
yhat = model.predict(image)
print(yhat)

# Convertir las probabilidades en etiquetas de clase 
label = decode_predictions(yhat, 100)
print(label)

# Recuperar el resultado más probable, p. ej. mayor probabilidad 
label = label[0][0]
print(label)

# Imprimir la clasificación
print('%s (%.2f%%)' % (label[1], label[2]*100))

# Eliminar la capa de salida 
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
model.summary()

# Obtener características extraídas 
features = model.predict(image)
print(features.shape)

# Guardar en archivo 
dump(features, open('dog.pkl', 'wb'))

