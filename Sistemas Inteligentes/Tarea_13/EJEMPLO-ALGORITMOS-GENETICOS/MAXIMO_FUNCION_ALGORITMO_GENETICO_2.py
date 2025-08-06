# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:33:00 2023

@author: Jaime
"""
#==============================================================================
# ENCONTRAR EL MÁXIMO DE UNA FUNCIÓN USANDO ALGORITMOS GENÉTICOS

# El código proporcionado implementa un algoritmo genético para maximizar una 
# función objetivo f(x)=x⋅sin(10⋅π⋅x)+2.0 en el intervalo [0, 2]. El algoritmo 
# representa las posibles soluciones como cromosomas binarios de longitud 20 y 
# los convierte a valores decimales que corresponden a puntos en el intervalo 
# [0, 2]. Se inicializa una población de 200 individuos, y a lo largo de 200 
# generaciones, se seleccionan padres con alta aptitud (es decir, aquellos 
# cuyos valores decimales maximizan la función objetivo), se cruzan para crear 
# descendencia y se aplica mutación aleatoria para mantener la diversidad 
# genética. La aptitud de cada individuo se mide evaluando la función objetivo 
# en el punto correspondiente al valor decimal mapeado. Después de evolucionar 
# la población a través de múltiples generaciones, el algoritmo identifica y 
# reporta el valor de x que maximiza la función objetivo y el valor máximo de 
# la función en ese punto.
#==============================================================================

# Cargar librerías
import numpy as np

# Función objetivo
def f(x):
    return x * np.sin(10 * np.pi * x) + 2.0

# Parámetros del algoritmo genético
tamaño_poblacion = 500
tasa_cruce = 0.9
tasa_mutacion = 0.02
num_generaciones = 300
longitud_cromosoma = 30  # Representación binaria

# Funciones auxiliares
def binario_a_decimal(binario):
    return binario.dot(2**np.arange(binario.size)[::-1])

def mapeo(x):
    # Mapea un número binario en el intervalo [0, 2]
    decimal_max = 2**longitud_cromosoma - 1
    return x * 2.0 / decimal_max

def aptitud(individuo):
    x = mapeo(binario_a_decimal(individuo))
    return f(x)

def seleccionar_padres(poblacion, aptitudes, num_padres):
    padres = np.empty((num_padres, poblacion.shape[1]))
    for padre in range(num_padres):
        max_aptitud_idx = np.where(aptitudes == np.max(aptitudes))
        max_aptitud_idx = max_aptitud_idx[0][0]
        padres[padre, :] = poblacion[max_aptitud_idx, :]
        aptitudes[max_aptitud_idx] = -999999
    return padres

def cruce(padre1, padre2):
    punto_cruce = np.random.randint(1, longitud_cromosoma - 1)
    hijo1 = np.hstack((padre1[:punto_cruce], padre2[punto_cruce:]))
    hijo2 = np.hstack((padre2[:punto_cruce], padre1[punto_cruce:]))
    return hijo1, hijo2

def mutacion(individuo):
    for idx in range(longitud_cromosoma):
        if np.random.rand() < tasa_mutacion:
            individuo[idx] = 1 - individuo[idx]
    return individuo

# Inicialización
poblacion = np.random.randint(2, size=(tamaño_poblacion, longitud_cromosoma))

# Evolución
for generacion in range(num_generaciones):
    aptitudes = np.apply_along_axis(aptitud, 1, poblacion)
    num_nuevos_descendientes = int(tasa_cruce * tamaño_poblacion)
    padres = seleccionar_padres(poblacion, aptitudes, num_nuevos_descendientes)
    
    descendencia = np.empty((num_nuevos_descendientes, longitud_cromosoma))
    k = 0
    while k < num_nuevos_descendientes:
        padre1, padre2 = padres[k], padres[k+1]
        hijo1, hijo2 = cruce(padre1, padre2)
        descendencia[k] = hijo1
        descendencia[k+1] = hijo2
        k += 2
    
    if tasa_mutacion > 0:
        for idx in range(num_nuevos_descendientes):
            if np.random.rand() < tasa_mutacion:
                descendencia[idx] = mutacion(descendencia[idx])
    
    poblacion[:num_nuevos_descendientes] = descendencia
    poblacion[num_nuevos_descendientes:] = seleccionar_padres(poblacion, aptitudes, tamaño_poblacion - num_nuevos_descendientes)

# Resultado final
aptitudes = np.apply_along_axis(aptitud, 1, poblacion)
mejor_solucion_idx = np.argmax(aptitudes)
mejor_solucion = poblacion[mejor_solucion_idx]
x_maximo = mapeo(binario_a_decimal(mejor_solucion))
print("x que maximiza f(x):", x_maximo, "Valor máximo de f(x):", f(x_maximo))
