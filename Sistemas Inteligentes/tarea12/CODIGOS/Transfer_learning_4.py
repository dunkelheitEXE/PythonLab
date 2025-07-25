# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 21:59:15 2021

@author: Jaime
"""
#==============================================================================
# En este código se construye un nuevo modelo de clasificación multiclase 
# basado en la arquitectura VGG16 preentrenada, excluyendo sus capas densas 
# originales (include_top=False) y adaptando su entrada a imágenes de tamaño 
# 300x300 píxeles. A la salida convolucional de VGG16 se le agrega una capa 
# Flatten para convertir los mapas de activación en un vector plano, seguida de 
# una capa densa intermedia con 1024 neuronas ReLU y una capa de salida softmax 
# con 100 clases, lo que configura una tarea de clasificación multiclase. 
# Luego, se redefine el modelo completo con la clase Model de Keras. Finalmente, 
# se congelan todas las capas del modelo —incluidas las nuevas— para evitar su 
# ajuste durante el entrenamiento, y se especifica explícitamente que ciertas 
# capas específicas del bloque 1 y 2 de VGG16 permanezcan no entrenables, lo 
# que permite aplicar transferencia de aprendizaje con control selectivo del 
# fine-tuning sobre capas profundas.
#==============================================================================

# Cargar librerías 
from keras.applications.vgg16 import VGG16
from keras.models import Model
from keras.layers import Dense
from keras.layers import Flatten

# Modelo sin capas clasificadoras 
model = VGG16(include_top=False, input_shape=(300, 300, 3))
model.summary()

# Agregar nuevas capas al clasificador  
flat1 = Flatten()(model.layers[-1].output)
class1 = Dense(1024, activation='relu')(flat1)
output = Dense(100, activation='softmax')(class1)

# Definir el nuevo modelo
model = Model(inputs=model.inputs, outputs=output)
model.summary()

# Marcar las capas cargadas como no entrenables 
for layer in model.layers:
	layer.trainable = False

# Marcar algunas capas como no entrenables
model.get_layer('block1_conv1').trainable = False
model.get_layer('block1_conv2').trainable = False
model.get_layer('block2_conv1').trainable = False
model.get_layer('block2_conv2').trainable = False


model.summary()






