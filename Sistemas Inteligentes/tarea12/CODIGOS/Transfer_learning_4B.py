# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 21:59:15 2021

@author: Jaime
"""
#==============================================================================
# En este código se construye un modelo personalizado a partir de la 
# arquitectura preentrenada VGG16, excluyendo sus capas densas originales 
# mediante el parámetro include_top=False para emplearlo como extractor de 
# características. A continuación, se encapsula dentro de un modelo Sequential 
# y se le agregan capas convolucionales adicionales con 64 filtros y max 
# pooling para continuar la extracción jerárquica de patrones espaciales. 
# Posteriormente, se añaden varias capas densas completamente conectadas para 
# realizar la clasificación final, culminando en una capa softmax con 100 
# salidas, lo que sugiere una tarea de clasificación multiclase. Este enfoque 
# representa una forma básica de transferencia de aprendizaje con expansión 
# personalizada, combinando el poder del aprendizaje previo con la adaptación 
# a un nuevo conjunto de clases específicas.
#==============================================================================

# Cargar librerías 
from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import  MaxPooling2D
from keras.models import Sequential

# Modelo sin capas clasificadoras 
model = VGG16()
model.summary()

model = Sequential()
model.add(VGG16(include_top=False))
model.summary()

# Agregar nuevas capas convolutivas al clasificador  
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.summary()

# Agregar nuevas capas densas al clasificador  
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(100, activation='softmax'))

# Definir el nuevo modelo
model.summary()










