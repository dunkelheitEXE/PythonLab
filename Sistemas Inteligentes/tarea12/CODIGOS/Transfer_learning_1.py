# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 20:24:32 2021

@author: Jaime
"""
#==============================================================================
# Este código implementa un sistema automatizado para realizar clasificación de 
# imágenes utilizando el modelo VGG19 preentrenado con ImageNet y aplicarlo 
# sobre un conjunto de imágenes almacenadas en un directorio local. Primero, 
# se importa el modelo VGG19 junto con las funciones necesarias para cargar y 
# preprocesar imágenes (load_img, img_to_array, preprocess_input) y para 
# decodificar las predicciones (decode_predictions). Se establece una excepción 
# (os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE") para evitar conflictos de 
# bibliotecas en ciertas configuraciones de CPU.

# Luego, se recorre un directorio con imágenes (image_dir), procesando hasta 
# 50 imágenes para evitar sobrecarga computacional. Para cada imagen, se 
# realiza el siguiente flujo: se carga y redimensiona a 224x224 píxeles (el 
# tamaño requerido por VGG19), se convierte a un arreglo NumPy, se expande su 
# dimensión para simular un batch, y se preprocesa conforme a los requisitos 
# del modelo. Posteriormente, se realiza la predicción de clases y se 
# decodifican las cuatro clases más probables con sus respectivas probabilidades, 
# las cuales se imprimen junto al nombre del archivo. Si ocurre un error con 
# alguna imagen (por ejemplo, si no es válida o está dañada), se captura la 
# excepción y se imprime el mensaje correspondiente, lo que garantiza que el 
# procesamiento continúe con las demás imágenes.

# En resumen, este código utiliza transferencia de conocimiento del modelo 
# VGG19 para etiquetar automáticamente un lote de imágenes nuevas con sus 
# clases más probables, implementando una solución eficaz de clasificación 
# visual basada en visión por computadora preentrenada.
#==============================================================================

# Cargar librerías y paquetes
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Cargar solo el modelo preentrenado
pretrained_model = VGG19(weights='imagenet')
pretrained_model.summary()

# Directorio con imágenes
image_dir = 'C:/Users/Asus.S510UNR/Desktop/MAYO-AGOSTO-2025/UPMH-MAYO-AGOSTO-2025/CODIGOS/CODIGOS-MAYO-AGOSTO-2025/SEMANA-4/sample'

# Iterar sobre las imágenes (solo las primeras 5 para evitar sobrecarga)
for i, file in enumerate(os.listdir(image_dir)):
    if i >= 50:
        break

    print(file)
    full_path = os.path.join(image_dir, file)

    try:
        # Preprocesamiento de imagen
        image = load_img(full_path, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # Predicción
        y_pred = pretrained_model.predict(image)
        label = decode_predictions(y_pred, top=4)[0]

        for class_id, name, prob in label:
            print(f"{name}: {prob:.4f}")
        print()

    except Exception as e:
        print(f"Error procesando la imagen {file}: {e}")
        print()
