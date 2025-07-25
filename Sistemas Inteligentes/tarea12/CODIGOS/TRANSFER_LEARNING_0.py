# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 19:42:19 2023

@author: Jaime
"""
#==============================================================================
# El código presentado implementa un sistema completo de clasificación binaria 
# de imágenes utilizando el modelo VGG19 preentrenado como extractor de 
# características congelado, con una arquitectura adicional personalizada y 
# entrenamiento sobre un conjunto de datos local. En primer lugar, se definen 
# las rutas hacia las carpetas que contienen las imágenes de entrenamiento y 
# validación, organizadas en subdirectorios por clase (por ejemplo, "cats" y 
# "dogs"). Luego se carga el modelo VGG19 sin sus capas clasificadoras 
# originales (include_top=False), preentrenado en ImageNet, y se congelan sus 
# pesos (trainable=False) para evitar su ajuste durante el entrenamiento. A 
# este bloque convolucional se le añaden una capa Flatten, una capa densa 
# intermedia con 256 neuronas y activación ReLU, y una capa de salida con una 
# sola neurona y activación sigmoide para realizar clasificación binaria.

# Se utilizan generadores de imágenes (ImageDataGenerator) para alimentar el 
# modelo, aplicando técnicas de aumento de datos (data augmentation) en el 
# conjunto de entrenamiento (rotaciones, desplazamientos, zoom, volteo, etc.) 
# y normalización por reescalado en ambos conjuntos. Las imágenes se cargan 
# desde directorios y se redimensionan a 150x150 píxeles, adaptando el tamaño 
# a la entrada esperada por VGG19. Posteriormente, se determina automáticamente 
# el número de pasos por época y de validación según el tamaño de los datasets 
# y el tamaño del batch. El modelo se entrena durante 30 épocas, evaluando su 
# rendimiento en el conjunto de validación, y finalmente se guarda el modelo 
# completo en el archivo 'cats_and_dogs_small_3.h5' para su reutilización 
# futura. En conjunto, este código ejemplifica el uso eficiente de la 
# transferencia de aprendizaje para tareas de clasificación de imágenes con un 
# dataset personalizado y limitado, combinando robustez, eficiencia y 
# adaptabilidad.
#==============================================================================

# Importar las bibliotecas necesarias
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import RMSprop

# Definir rutas a los directorios de entrenamiento y validación
train_dir = 'C:/Users/Asus.S510UNR/Desktop/MAYO-AGOSTO-2025/UPMH-MAYO-AGOSTO-2025/CODIGOS/CODIGOS-MAYO-AGOSTO-2025/SEMANA-4/dataset/training_set'
validation_dir = 'C:/Users/Asus.S510UNR/Desktop/MAYO-AGOSTO-2025/UPMH-MAYO-AGOSTO-2025/CODIGOS/CODIGOS-MAYO-AGOSTO-2025/SEMANA-4/dataset/test_set'

# Cargar la red VGG19 preentrenada
conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
conv_base.trainable = False  # Congelar las capas convolucionales

# Construir el modelo
model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Para clasificación binaria

# Compilar el modelo
model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(learning_rate=2e-5),
    metrics=['acc']
)

# Crear generadores de datos con aumento para entrenamiento y sin aumento para validación
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Cargar las imágenes desde los directorios
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

# Calcular steps_per_epoch y validation_steps automáticamente
steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = validation_generator.samples // validation_generator.batch_size

# Entrenar el modelo
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

# Guardar el modelo
model.save('cats_and_dogs_small_3.h5')
