# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 20:24:32 2021

@author: Jaime
"""
# Cargar librerías y paquetes
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Crear un objeto con el modelo VGG16 previamente entrenado.
model = VGG19()

# Resumen del modelo
model.summary()

# Definir un modelo nuevo con pesos aleatorios y 10 clases
new_input = Input_shape=(640, 480, 3)
model = VGG19(weights=None, input_shape=new_input, classes=10)
# Resumen del modelo
model.summary()


    
    
    
    
    
    

    