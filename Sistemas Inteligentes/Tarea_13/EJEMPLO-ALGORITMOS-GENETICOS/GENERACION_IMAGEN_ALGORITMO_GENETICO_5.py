# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 11:17:58 2023

@author: Jaime
"""
#==============================================================================
# GENERACIÓN DE IMÁGENES SIMILARES A OTRAS USANDO ALGORITMOS GENÉTICOS

# El código implementa un algoritmo genético para aproximar una imagen objetivo, 
# en este caso un mapa de México, mediante la superposición de rectángulos de 
# diferentes tamaños, posiciones y colores. El objetivo es hacer que la imagen 
# generada por la superposición de estos rectángulos se asemeje lo más posible 
# a la imagen objetivo. Inicialmente, se genera una población aleatoria de 
# individuos, donde cada individuo representa una configuración de 100 
# rectángulos definidos por su posición, tamaño y color. A lo largo de 500 
# generaciones, los individuos se evalúan en función de cuán bien su imagen 
# generada se aproxima a la imagen objetivo (medido por la diferencia 
# cuadrática entre ambas imágenes). Los mejores individuos se seleccionan 
# para reproducirse, generando descendencia a través de cruces y mutaciones 
# aleatorias. La mejor solución encontrada se actualiza y visualiza 
# periódicamente, y al final del proceso, se muestra la imagen generada que 
# más se asemeja a la imagen objetivo. Este proceso es una demostración de 
# cómo los algoritmos genéticos pueden ser utilizados para resolver problemas 
# de optimización en la generación de imágenes.
#==============================================================================

# Cargar librerías
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Parámetros del problema
num_rectangulos = 100
tamaño_imagen = (100, 100)  # Cambia esto para que coincida con la resolución de tu imagen

# Parámetros del algoritmo genético
tamaño_poblacion = 100
tasa_mutacion = 0.01
num_generaciones = 500

# Funciones auxiliares
def generar_poblacion_inicial():
    # Ahora cada rectángulo tiene una posición (x, y), ancho, alto y un color RGBA
    return np.random.rand(tamaño_poblacion, num_rectangulos, 7)

def calcular_aptitud(individuo, imagen_objetivo):
    imagen = np.zeros_like(imagen_objetivo)
    imagen = agregar_rectangulo(imagen, individuo)  # Pasar el individuo completo
    return -np.sum((imagen - imagen_objetivo)**2)

def seleccionar_mejores(poblacion, aptitudes, num_seleccionados):
    indices = np.argsort(aptitudes)
    return poblacion[indices[-num_seleccionados:]]

def cruzar(padre1, padre2):
    punto_cruce = np.random.randint(num_rectangulos)
    return np.vstack((padre1[:punto_cruce], padre2[punto_cruce:]))

def mutar(individuo):
    for i in range(num_rectangulos):
        if np.random.rand() < tasa_mutacion:
            # Mutamos todos los atributos del rectángulo
            individuo[i] = np.random.rand(7)
    return individuo

def agregar_rectangulo(imagen, individuo):
    fig, ax = plt.subplots(1, figsize=(tamaño_imagen[0] / 80, tamaño_imagen[1] / 80), dpi=80)
    ax.imshow(imagen)
    for rectangulo in individuo:
        x, y, ancho, alto, r, g, b = rectangulo
        color = (r, g, b, 1)  # El último valor es el alpha (transparencia)
        ax.add_patch(patches.Rectangle((x * tamaño_imagen[0], y * tamaño_imagen[1]), ancho * tamaño_imagen[0], alto * tamaño_imagen[1], color=color))
    fig.canvas.draw()
    imagen = np.array(fig.canvas.renderer._renderer)
    plt.close(fig)
    if imagen.shape[-1] == 4:
        imagen = imagen[..., :3]  # Solo utilizamos los tres primeros canales
    return imagen

# Algoritmo genético
imagen_objetivo = Image.open('C:/Users/Asus.S510UNR/Desktop/EJEMPLO-ALGORITMOS-GENETICOS/MAPA_MEXICO.jpg')  # Reemplazar con la ruta a tu imagen
imagen_objetivo = imagen_objetivo.resize(tamaño_imagen)
imagen_objetivo = np.array(imagen_objetivo)
poblacion = generar_poblacion_inicial()
mejor = None
mejor_aptitud = -np.inf

for _ in range(num_generaciones):
    aptitudes = np.array([calcular_aptitud(individuo, imagen_objetivo) for individuo in poblacion])
    if np.max(aptitudes) > mejor_aptitud:
        mejor = poblacion[np.argmax(aptitudes)]
        mejor_aptitud = np.max(aptitudes)
        plt.imshow(agregar_rectangulo(np.zeros_like(imagen_objetivo), mejor))
        plt.show()
    nueva_poblacion = seleccionar_mejores(poblacion, aptitudes, tamaño_poblacion // 2)
    for _ in range(tamaño_poblacion // 2):
        padre1, padre2 = np.random.choice(len(nueva_poblacion), 2, replace=False)
        hijo = cruzar(nueva_poblacion[padre1], nueva_poblacion[padre2])[np.newaxis, :]
        nueva_poblacion = np.vstack((nueva_poblacion, hijo))

    if np.random.random() < tasa_mutacion:
        indice = np.random.randint(tamaño_poblacion)
        nueva_poblacion[indice] = mutar(nueva_poblacion[indice])

    poblacion = nueva_poblacion

print('Mejor solución encontrada:')
plt.imshow(agregar_rectangulo(np.zeros_like(imagen_objetivo), mejor))
plt.show()
