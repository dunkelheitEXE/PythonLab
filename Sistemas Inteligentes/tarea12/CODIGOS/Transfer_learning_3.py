# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 23:12:37 2021

@author: Jaime
"""
#==============================================================================
# El código presentado muestra un ejemplo práctico del uso del modelo VGG16 
# preentrenado para realizar extracción de características (feature extraction) 
# a partir de una imagen individual. El proceso comienza cargando una imagen 
# (dog.jpg) desde el disco y redimensionándola al tamaño esperado por VGG16 
# (224×224 píxeles). Luego, la imagen se convierte en un arreglo NumPy y se 
# reestructura para cumplir con el formato de entrada del modelo 
# ((1, 224, 224, 3)), aplicando posteriormente el preprocesamiento necesario 
# (preprocess_input) para normalizar los valores de píxeles según los 
# estándares de VGG16. A continuación, se carga el modelo VGG16 completo con 
# sus pesos preentrenados en ImageNet, y se redefine el modelo para eliminar 
# la capa de salida softmax, de modo que la salida del nuevo modelo sea el 
# vector de activación de la penúltima capa densa (fc2), que contiene una 
# representación numérica profunda y de alta abstracción de la imagen. 
# Finalmente, esta representación de características se calcula y se guarda en 
# un archivo dog.pkl mediante pickle, lo que permite utilizarla posteriormente 
# en otras tareas como clasificación personalizada, clustering, recuperación 
# de imágenes o análisis semántico. En suma, este código implementa un flujo 
# completo para convertir una imagen en una representación vectorial profunda 
# reutilizable.
#==============================================================================

# Cargar librerías
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import Model
from pickle import dump

# Almacenar una imagen de archivo
image = load_img('C:/Users/Asus.S510UNR/Desktop/MAYO-AGOSTO-2025/UPMH-MAYO-AGOSTO-2025/CODIGOS/CODIGOS-MAYO-AGOSTO-2025/SEMANA-4/dog.jpg', target_size=(224, 224))

# Convertir la imagen en pixeles a un arreglo de numpy
image = img_to_array(image)

# Cambiar o reformular la imagen para ser usada en el modelo
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))

# Preparar la imagen para el modelo VGG 
image = preprocess_input(image)

# Almacenar el modelo
model = VGG16()

# Resumen del modelo
model.summary()

# Remover la capa de salida
model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

# Extraer las características
features = model.predict(image)
print(features.shape)

# Salvar el archivo
dump(features, open('dog.pkl', 'wb'))