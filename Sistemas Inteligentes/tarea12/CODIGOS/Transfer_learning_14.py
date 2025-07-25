# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 23:07:46 2025

@author: Asus.S510UNR
"""

# =======================
# 1. Cargar librerías
# =======================
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# =======================
# 2. Cargar y preparar datos
# =======================
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# =======================
# 3. Crear generador personalizado con resize por lotes
# =======================
class CIFAR10DataGenerator(Sequence):
    def __init__(self, images, labels, batch_size=64, is_training=True):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.is_training = is_training
        self.indices = np.arange(len(images))

    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = [tf.image.resize(self.images[i], [224, 224]) for i in batch_indices]
        batch_images = preprocess_input(tf.stack(batch_images))
        batch_labels = self.labels[batch_indices]
        return batch_images, batch_labels

    def on_epoch_end(self):
        if self.is_training:
            np.random.shuffle(self.indices)

# =======================
# 4. Generadores de entrenamiento y validación
# =======================
train_generator = CIFAR10DataGenerator(x_train, y_train_cat, batch_size=64, is_training=True)
test_generator = CIFAR10DataGenerator(x_test, y_test_cat, batch_size=64, is_training=False)

# =======================
# 5. Cargar base VGG19 sin top
# =======================
conv_base = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
conv_base.trainable = False  # Congelar pesos

# =======================
# 6. Construir modelo completo
# =======================
model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

# =======================
# 7. Compilar modelo
# =======================
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

# =======================
# 8. Entrenar modelo
# =======================
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=test_generator,
    verbose=1
)

# =======================
# 9. Visualizar métricas
# =======================
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()

# =======================
# 10. Guardar modelo entrenado
# =======================
model.save('vgg19_cifar10_transfer.h5')

