# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 20:24:32 2021

@author: Jaime
"""
# Cargar librerías y paquetes
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

# Crear un objeto con el modelo VGG19 previamente entrenado.
model = VGG19(weights='imagenet')

# Resumen del modelo
model.summary()

# Ruta al directorio de imágenes
image_dir = 'C:/Users/Asus.S510UNR/Desktop/MAYO-AGOSTO-2025/UPMH-MAYO-AGOSTO-2025/CODIGOS/CODIGOS-MAYO-AGOSTO-2025/SEMANA-4/sample'

# Iterar sobre cada imagen en el directorio
for file in os.listdir(image_dir):
    print(file)

    # Construcción segura de la ruta completa al archivo
    full_path = os.path.join(image_dir, file)

    try:
        # Cargar y preprocesar la imagen
        image = load_img(full_path, target_size=(224, 224))
        image = img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = preprocess_input(image)

        # Predecir con el modelo VGG19
        y_pred = model.predict(image)

        # Decodificar y mostrar las 3 clases más probables
        label = decode_predictions(y_pred, top=3)[0]
        for class_id, name, prob in label:
            print(f"{name}: {prob:.4f}")
        print()

    except Exception as e:
        print(f"Error procesando la imagen {file}: {e}")
        print()
